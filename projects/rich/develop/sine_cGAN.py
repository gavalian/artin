import torch
from torch import nn
import math
import matplotlib.pyplot as plt

n_classes = int(1)
n_input   = int(2)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, noise,labels):
        x = torch.cat((labels, noise), -1)
        #x = torch.cat((labels, noise))
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            #nn.Linear(32, 32),
            #nn.ReLU(),
            nn.Linear(32, 2))

    def forward(self, noise, labels):
        x = torch.cat((labels,noise),-1)
        output = self.model(x)
        return output

train_data_length = 1024*24

#train_labels = torch.zeros(train_data_length)
train_labels = torch.rand(train_data_length,1)*0.8+0.2
train_data   = torch.zeros((train_data_length, 2))
#train_data[:, 0] = torch.rand(train_data_length)
train_data[:, 0] = torch.rand(train_data_length)
train_data[:, 1] = train_labels[:,0]*torch.sin(2*math.pi*train_data[:, 0])*0.5+0.5
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]
#print(train_data)
#print(train_labels)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)

discriminator = Discriminator()
generator = Generator()

data = generator(train_data,train_labels)

print(train_labels)
print(train_data)

print(data)
plt.plot(train_data[:, 0], train_data[:, 1], ".")
#plt.pause(5000)

#lr = 0.001
lr = 0.001
num_epochs = 420
loss_function = nn.BCELoss()


optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, real_labels) in enumerate(train_loader):
        # Data for descriminator training
        real_samples_labels = real_labels#torch.ones((batch_size, 1))
        latent_space_samples = torch.rand((batch_size, 2))
        latent_space_labels  = torch.rand((batch_size, 1))*0.8+0.2
        #print(latent_space_labels)
        #print(latent_space_samples)
        #print(real_samples_labels)
        generated_samples = generator(latent_space_samples,latent_space_labels)        
        #generated_samples_labels = #torch.zeros((batch_size, 1))

        all_samples = torch.cat((real_samples, generated_samples))
        
        latent_labels_ONE = torch.zeros((batch_size,1));
        real_labels_ONE = torch.ones((batch_size,1));
        
        all_samples_labels = torch.cat(
            (real_samples_labels, latent_space_labels))

        all_labels_ONE = torch.cat((real_labels_ONE,latent_labels_ONE))

        #print(all_samples_labels)
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples, all_samples_labels)

        loss_discriminator = loss_function(
            output_discriminator, all_labels_ONE) # fixed all
        
        loss_discriminator.backward()
        optimizer_discriminator.step()

        lss01 = torch.rand((batch_size, 2))
        lsl01 = torch.rand((batch_size, 1))*0.8+0.2

        generator.zero_grad()
        generated_samples = generator(lss01,lsl01)
        output_discriminator_generated = discriminator(generated_samples,lsl01)
        loss_generator = loss_function(
            output_discriminator_generated, real_labels_ONE)
        loss_generator.backward()
        optimizer_generator.step()
        #print('output:')
        #print(generated_samples)
        # Output value of loss function
        if epoch % 2 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")     
            eval_labels = torch.rand(300,1)*0.8+0.2
            eval_pool = torch.rand(300,2)
            eval_labels[:,0] = 0.8
            generated = generator(eval_pool, eval_labels)
            generated = generated.detach().numpy()  
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.plot(generated[:, 0], generated[:, 1], ".")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0,1.0])                                     
            fig.savefig('output/output_'+str(epoch)+".png")
            plt.close(fig)
            


torch.save(generator.state_dict(), 'model_conditional_ring.torch')
eval_labels = torch.rand(300,1)*0.8+0.2
eval_pool = torch.rand(300,2)

eval_labels[:,0] = 0.8

generated = generator(eval_pool, eval_labels)
generated = generated.detach().numpy()
print(generated)
plt.plot(generated[:, 0], generated[:, 1], ".")
plt.pause(5000)