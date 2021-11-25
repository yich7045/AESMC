import torch
from AESMC import AESMC
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# define the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 1
# manual seed
torch.manual_seed(seed)

# hyperparameters
action_space = 2
nr_inputs = 1
observation_type = '84x84'
action_encoding = 10
cnn_channels = [8, 16, 8]
h_dim = 100
init_function = 'orthogonal_'
encoder_batch_norm = False
policy_batch_norm = False
prior_loss_coef = 1.0
obs_loss_coef = 1.0
detach_encoder = False
batch_size = 1
num_particles = 20
particle_aggregation = 'rnn'
z_dim = 128
resample = True
predicted_times=list(range(65))
learning_rate = 2e-03
save_every = 5

DVRL_model = AESMC(action_space,
              nr_inputs,
              observation_type,
              action_encoding,
              cnn_channels,
              h_dim,
              init_function,
              encoder_batch_norm,
              policy_batch_norm,
              prior_loss_coef,
              obs_loss_coef,
              detach_encoder,
              batch_size,
              num_particles,
              particle_aggregation,
              z_dim,
              resample)

DVRL_model.to(device)
optimizer = torch.optim.RMSprop(DVRL_model.parameters(), lr=learning_rate)

masks = torch.FloatTensor([0.0, 1.0])
masks = masks.to(device)

def train(epoch, train_loader):
    epoch_loss = 0
    current_memory = {
        'current_obs': None,
        'oneHotActions': torch.FloatTensor([[[1,1]]]),
        'states': DVRL_model.new_latent_state(),
        'predicted_time': None
    }
    for batch_index, (sequence_data, sequence_target) in enumerate(train_loader):
        train_loss = torch.tensor([0.0])
        train_loss = train_loss.to(device)
        current_memory['states'] = DVRL_model.vec_conditional_new_latent_state(current_memory['states'], masks[0])
        print(current_memory['states'])
        print(current_memory['states'].h.size())
        print(current_memory['states'].log_weight.size())
        for i in range(sequence_data.size()[1]):
            data = sequence_data[0][i]
            data = data.view(1, 1, data.size()[0], data.size()[1])
            data = data.float()
            data = data / 255.
            current_memory['current_obs'] = data
            policy_return = DVRL_model(current_memory, deterministic = False, predicted_times = None)
            loss = policy_return.total_encoding_loss
            train_loss += loss
            epoch_loss += loss.item()
            current_memory['states'] = DVRL_model.vec_conditional_new_latent_state(policy_return.latent_state, masks[1])
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        current_memory['states'] = current_memory['states'].detach()
    num_train_batches = len(train_loader)
    show_loss = epoch_loss / num_train_batches
    if (epoch % save_every == 0):
        print('\tEpoch : {} Model Train Loss: {:.3f}'.format(epoch, show_loss))
    return show_loss


def test(epoch, test_loader, predicted_times):
    test_current_memory = {
        'current_obs': None,
        'oneHotActions': torch.FloatTensor([[[1,1]]]),
        'states': DVRL_model.new_latent_state(),
        'predicted_time': predicted_times
    }
    for batch_index, (sequence_data, sequence_target) in enumerate(test_loader):
        data = sequence_data[0][0]
        data = data.view(1, 1, 84, 84)
        data = data.float()
        data = data.to(device)
        data = data / 255.
        test_current_memory['current_obs'] = data.float()
        policy_return = DVRL_model(test_current_memory, deterministic=False, predicted_times=predicted_times)
        for i in range(len(policy_return.predicted_obs_img)):
            if batch_index == 10:
                test_img = policy_return.predicted_obs_img[i][0][0].cpu()
                test_img = test_img.detach().numpy()
                plt.imshow(test_img)
                plt.savefig('save_img1/' + str(batch_index) + '_' + str(epoch) + '_' + str(i) +'.png')


class Looping_data(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

train_img = np.load('84x84_AESMC_train.npz')
train_Data = train_img['arr_0']
train_Data = np.reshape(train_Data, (-1, 65, 84, 84))

test_img = np.load('84x84_AESMC_test.npz')
test_Data = test_img['arr_0']
test_Data = np.reshape(test_Data, (-1, 65, 84, 84))
test_dataset = Looping_data(test_Data)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

n_epochs = 5000
for epoch in range(0, n_epochs+1):
    train_Data = shuffle(train_Data)
    train_dataset = Looping_data(train_Data)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    train_loss = train(epoch, train_loader)
    if epoch%save_every == 0:
        test_loss = test(epoch, test_loader, predicted_times)
        fn = 'model_save/AESMC_fully_observable_test1+' + str(epoch) + '.pth'
        torch.save(DVRL_model.state_dict(), fn)
        print('Saved model to ' + fn)