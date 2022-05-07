class Parameters():
  def  __init__(self,  language='hi',encoder_layers=1,decoder_layers=1,embedding_dim=128,\
                layer_type='lstm', units=128, dropout=0.5, attention=False,batch_size=128,\
                apply_beam_search=False,apply_teacher_forcing=True,\
                 save_outputs=None,epochs=5,wandb=False,beamWidth=5,restoreBestModel=True,\
                 patience=2):
        self.language = language
        self.embedding_dim = embedding_dim
        self.encoder_layers=encoder_layers
        self.decoder_layers=decoder_layers
        self.layer_type = layer_type
        self.units = units
        self.dropout = dropout
        self.attention = attention
        self.stats = []
        self.wandb=wandb
        self.epochs=epochs
        self.batch_size = 128
        self.apply_beam_search = apply_beam_search
        self.batch_size = batch_size
        self.apply_teacher_forcing=apply_teacher_forcing
        self.save_outputs=save_outputs
        self.restoreBestModel=restoreBestModel
        self.patience=patience