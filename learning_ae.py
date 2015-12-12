import data
import deep
import deep.autoencoder as ae
import utils

def create_autoencoder(in_path,out_path,dim=0):
    actions=data.read_actions(action_path)
    imgs=data.get_projections(dim,actions)
    cls=ae.built_ae_cls()
    deep.learning_iter_unsuper(cls,imgs,n_epochs=500)
    utils.save_object(cls.model,out_path) 

if __name__ == "__main__":
    action_path="../_final_actions/"
    ae_path="../nn/zy_ae"
    create_autoencoder(action_path,ae_path,2)

