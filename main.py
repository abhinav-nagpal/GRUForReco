class SessionDataset: # sep
    def __init__(self, data, sessionColumn='SessionId', itemColumn='ItemId', timeColumn='Time', n_samples=-1, map=None):
  
        self.df = data
        self.sessionColumn = sessionColumn
        self.itemColumn = itemColumn
        self.timeColumn = timeColumn
        self.joinIndices(map=map)
        self.df.sort_values([sessionColumn, timeColumn], inplace=True)

        self.addclicks = self.addClicks()
        self.session_idx_arr = self.orderSession()
        
    def addClicks(self):
        
        add = np.zeros(self.df[self.sessionColumn].nunique() + 1, dtype=np.int32)
        add[1:] = self.df.groupby(self.sessionColumn).size().cumsum()

        return add

    def orderSession(self):
        sort_time = False
        sessions = np.arange(self.df[self.sessionColumn].nunique())

        return sessions
    
    def joinIndices(self, map=None):

        if map is None:
            item_ids = self.df[self.itemColumn].unique()  # unique item ids
            item2idx = pd.Series(data=np.arange(len(item_ids)),
                                 index=item_ids)
            map = pd.DataFrame({self.itemColumn:item_ids,
                                   'item_idx':item2idx[item_ids].values})
        
        
        self.map = map
        self.df = pd.merge(self.df, self.map, on=self.itemColumn, how='inner')
        
    @property    
    def items(self):
        return self.map.ItemId.unique()
        

class SessionDataLoader:
    def __init__(self, dataset, batch_size=50):

        self.dataset = dataset
        self.batch_size = batch_size
        self.done_sessions_counter = 0
        
    def __iter__(self):

        df = self.dataset.df
        sessionColumn='SessionId'
        itemColumn='ItemId'
        timeColumn='TimeStamp'
        self.n_items = df[itemColumn].nunique()+1
        addclicks = self.dataset.addclicks
        session_idx_arr = self.dataset.session_idx_arr

        iterations = np.arange(self.batch_size)
        num_iterations = iterations.max()
        s = addclicks[session_idx_arr[iterations]]
        e = addclicks[session_idx_arr[iterations] + 1]
        mk = [] 
        done = False        

        while not done:
            minlen = (e - s).min()
            idx_target = df.item_idx.values[s]
            for i in range(minlen - 1):
                idx_input = idx_target
                idx_target = df.item_idx.values[s + i + 1]
                inp = idx_input
                target = idx_target
                yield inp, target, mk
                
            s = s + (minlen - 1)
            mk = np.arange(len(iterations))[(e - s) <= 1]
            self.done_sessions_counter = len(mk)
            for i in mk:
                num_iterations += 1

                if num_iterations >= len(addclicks) - 1:
                    done = True
                    break

                iterations[i] = num_iterations
                s[i] = addclicks[session_idx_arr[num_iterations]]
                e[i] = addclicks[session_idx_arr[num_iterations] + 1]

def construct_model(bs, tr_num_items,  lr, hidden_units, dropout_val):   

    inp = Input(batch_shape=(bs, 1, tr_num_items))
    gru, _ = GRU(hidden_units, stateful=True, return_state=True, name="GRU")(inp)
    drop = Dropout(dropout_val)(gru)
    preds = Dense(tr_num_items, activation='softmax')(drop)
    model = Model(inputs=inp, outputs=[preds])
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    model.compile(loss=categorical_crossentropy, optimizer=opt)
    model.summary()

    filepath='./model_checkpoint.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
    
    return model

def train(model, data_tr, data_val, data_tt, tr_num_items, bs, ep, tr_num_sessions, save_weights, eval_all_epochs):
    
    tr_dataset = SessionDataset(data_tr)
    batch_size = bs

    for epoch in range(ep):
        with tqdm(total=tr_num_sessions) as bar:
            loader = SessionDataLoader(tr_dataset, batch_size=bs)
            for feature, targ, mk in loader:

                gru_layer = model.get_layer(name="GRU")
                hidden_states = gru_layer.states[0].numpy()
                for i in mk:
                    hidden_states[i, :] = 0
                gru_layer.reset_states(states=hidden_states)

                x = to_categorical(feature, num_classes=loader.n_items)
                x = np.expand_dims(x, axis=1)

                y = to_categorical(targ, num_classes=loader.n_items)

                tr_loss = model.train_on_batch(x, y)

                bar.set_description("Epoch {0}. Loss: {1:.2f}".format(epoch, tr_loss))
                bar.update(loader.done_sessions_counter)

        if save_weights == True:
            model.save('./GRUForReco_{}.h5'.format(epoch))

        if eval_all_epochs == True:
            (rec, rec_k), (mrr, mrr_k) = getMetrics(model, data_tr, data_tt, bs, tr_num_items, tr_dataset.map)
            print("\t - Recall@{} epoch {}: {:2f}".format(rec_k, epoch, rec))
            print("\t - MRR@{}    epoch {}: {:2f}\n".format(mrr_k, epoch, mrr))

    if not eval_all_epochs:
        (rec, rec_k), (mrr, mrr_k) = getMetrics(model, data_tr, data_tt, bs, tr_num_items, tr_dataset.map)
        print("\t - Recall@{} epoch {}: {:2f}".format(rec_k, epochs, rec))
        print("\t - MRR@{}    epoch {}: {:2f}\n".format(mrr_k, epochs, mrr))

def getMetrics(model, data_tr, data_tt, bs, tr_num_items, tr_generator, recall_k=20, mrr_k=20):

    tt_dataset = SessionDataset(data_tt, map=tr_generator)
    tt_generator = SessionDataLoader(data_tt, batch_size=bs)
    mrr_sum = 0
    k = 0
    rec_sum = 0

    for feature, lab, mk in tt_generator:
        gru_unit = model.get_layer(name="GRU")
        hidden_states = gru_unit.states[0].numpy()
        for i in mk:
            hidden_states[i, :] = 0
        gru_unit.reset_states(states=hidden_states)
        x  = to_categorical(feature,  num_classes=tr_num_items)
        x = np.expand_dims(x, axis=1)
        y = to_categorical(lab, num_classes=tr_num_items)
        preds = model.predict(x, batch_size=bs)
        for idx in range(feature.shape[0]):
            pred_id = preds[idx]
            label_id = y[idx]
            rec =  pred_id.argsort()[-recall_k:][::-1]
            mrr =  pred_id.argsort()[-mrr_k:][::-1]
            tru = label_id.argsort()[-1:][::-1]
            k+=1
            if tru[0] in mrr:
                mrr_sum += 1/int((np.where(mrr == tru[0])[0]+1))
            if tru[0] in rec:
                rec_sum += 1
    recall = rec_sum/k
    mrr = mrr_sum/k
    return (recall, recall_k), (mrr, mrr_k)

if __name__ == '__main__':
  
  train_file = './data_processed/rsc15_train_tr.txt'
  do_eval = False
  val_file = './data_processed/rsc15_train_valid.txt'
  test_file = './data_processed/rsc15_test.txt'
  bs = 512
  save_weights = False
  ep = 10
  eval_all_epochs = True
  restart = False
  hidden_units = 100
  dropout_val=0.25
  lr = 0.001

  data_tr = pd.read_csv(train_file, sep='\t', dtype={'ItemId': np.int64})
  data_val   = pd.read_csv(val_file,   sep='\t', dtype={'ItemId': np.int64})
  data_tt  = pd.read_csv(test_file,  sep='\t', dtype={'ItemId': np.int64})

  tr_num_items = len(data_tr['ItemId'].unique()) + 1

  tr_num_sessions = len(data_tr['SessionId'].unique()) + 1
  tt_num_sessions = len(data_tt['SessionId'].unique()) + 1

  model = construct_model(bs, tr_num_items, lr, hidden_units, dropout_val)
  
  train(model, data_tr, data_val, data_tt, tr_num_items, bs, ep, tr_num_sessions, save_weights, eval_all_epochs)