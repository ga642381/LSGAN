import numpy as np
import torch
import torch.nn as nn
import utils
import cv2
import os
import time
from PIL import Image

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchsummary import summary

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

Data_Train = utils.load_training_data("./faces")
Data_Train = [np.swapaxes(image, 0, 2) for image in Data_Train]
Data_Train = [np.swapaxes(image, 1, 2) for image in Data_Train]
Data_Train = torch.stack([torch.Tensor(i) for i in Data_Train])
Data_Train = TensorDataset(Data_Train)


        
class Generator(nn.Module):
    def __init__(self, noise_dim=100, input_size=65):
        super().__init__()
        self.noise_dim = noise_dim
        self.input_size = input_size
        
        
        
        self.fc = nn.Sequential(                
                nn.Linear(self.noise_dim, 5*5*128),
                nn.BatchNorm1d(5*5*128)                
                )
        
        self.deconvs = nn.Sequential(                
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),            
                nn.ReLU(),
                
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),            
                nn.ReLU(),
                
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                nn.ConvTranspose2d(32, 3,kernel_size=3, stride=1, padding=1),                               
                nn.Tanh()
                 )
        utils.initialize_weights(self)
    
    def forward(self, z):
        output = self.fc(z)
        output = output.view(-1, 128, 5, 5) # (batch_size, channel, , )
        output = self.deconvs(output)
        return output
    
    
    
    
class Discriminator(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, input_size=65):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        
        self.convs = nn.Sequential(
                nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                nn.Conv2d(128,256, kernel_size=5, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
                )
        self.fc = nn.Sequential(
                nn.Linear(256*3*3, 1)
                )
        utils.initialize_weights(self)
       
    def forward(self,i):
        output = self.convs(i)
        output = output.view(-1, output.shape[1] * output.shape[2] * output.shape[3])
        output = self.fc(output)
        return output
        

class LSGAN(object):
    def __init__(self, Data_Train):
        self.epoch = 50000
        self.batch_size = 25
        self.noise_dim = 30
        
        # load datasets
        self.dataloader = DataLoader(dataset=Data_Train, batch_size=self.batch_size, shuffle=True)
        
        
        
        #network init
        self.G = Generator(noise_dim=self.noise_dim, input_size=65)
        self.D = Discriminator(input_dim=3, input_size=65)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=0.00004)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.00004)
        self.G = self.G.to(device)
        self.D = self.D.to(device)
        self.LS_loss = nn.MSELoss().to(device)
        
        print('---------- Networks architecture -------------')
        summary(self.G, (self.noise_dim,))
        summary(self.D, (3,65,65))
        print('-----------------------------------------------')
        
        
        # fixed noise
        torch.manual_seed(10)
        self.sample_noise = torch.rand((self.batch_size, self.noise_dim))
        self.sample_noise = self.sample_noise.to(device)
        
    
    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        self.y_real_ = self.y_real_.to(device)
        self.y_fake_ = self.y_fake_.to(device)
        
        
        print("Training Start !!!")
        total_start_time = time.time()
        self.D.train()
        for epoch in range(self.epoch):
            epoch_start_time = time.time()
            self.G.train()
            for step, batch_x in enumerate(self.dataloader): 
                
                if step == self.dataloader.dataset.__len__() // self.batch_size:
                    break
                
                batch_x = batch_x[0]
                batch_noise = torch.rand((self.batch_size, self.noise_dim))
                batch_x = batch_x.cuda()
                batch_noise = batch_noise.cuda()
                
                #===== train D =====#
                #self.D.train()
                #self.G.eval()
                
                self.D_optimizer.zero_grad() #gradient initialization
                D_real = self.D(batch_x)
                D_real_loss = self.LS_loss(D_real, self.y_real_)
                D_real_loss.backward()
                
                
                G_output = self.G(batch_noise)
                D_fake = self.D(G_output.detach())
                D_fake_loss = self.LS_loss(D_fake, self.y_fake_)
                D_fake_loss.backward()
                
                D_total_loss = D_real_loss + D_fake_loss
                self.train_hist["D_loss"].append(D_total_loss.item())
                
                '''
                print("---------------------train D ------------------------")
                if self.D.training:
                    print("D is training")
                else:
                    print("D is eval")
                if self.G.training:
                    print("G is training")
                else:
                    print("G is eval")
                
                '''
                self.D_optimizer.step()
                
                
                #===== train G =====#
                #self.D.eval()
                #self.G.train()
                
                self.G_optimizer.zero_grad()
                G_output = self.G(batch_noise)
                D_fake = self.D(G_output)
                G_loss = self.LS_loss(D_fake, self.y_real_)
                self.train_hist["G_loss"].append(G_loss.item())
                
                G_loss.backward()
                
                '''
                print("---------------------train G ------------------------")
                if self.D.training:
                    print("D is training")
                else:
                    print("D is eval")
                if self.G.training:
                    print("G is training")
                else:
                    print("G is eval")
                '''
                
                self.G_optimizer.step()
                
                #===== print some info =====#
                if ((step + 1) % 100) == 0:
                    print("Epoch: [{}] [{}/{}] D_loss: {}, G_loss: {}".format(
                           (epoch + 1),
                           (step + 1),
                           self.dataloader.dataset.__len__() // self.batch_size, 
                           D_total_loss.item(),
                           G_loss.item()
                           )
                          )
                
            self.train_hist["per_epoch_time"].append(time.time() - epoch_start_time)
            

                
                
            if epoch % 10 == 0:
                print("epoch : ", epoch)
                print("saving model...")
                with torch.no_grad():
                    self.visualize_results((epoch + 1))
                self.save(epoch + 1)
        self.train_hist['total_time'].append(time.time() - total_start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
          self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
        
        
    
    def visualize_results(self, epoch, fix=True):
        self.G.eval()
        self.result_dir = "LSGAN_result"
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        if fix:
            samples = self.G(self.sample_noise)
        
        else:
            sample_noise = torch.rand((self.batch_size, self.noise_dim)) # [0,1)
            sample_noise = sample_noise.to(device)
            
            samples = self.G(sample_noise)
            
        samples = samples.to(torch.device("cpu")).data.numpy().transpose(0, 2, 3, 1)
        samples = ((samples + 1) / 2) * 255        
        results = samples[:25].clip(0,255).astype('uint8')
        Images = []
        for result in results:
            Images.append(Image.fromarray(result).resize((64,64)))
            
        x_offsets = range(0, 64*4+1,64)
        y_offsets = range(0, 64*4+1,64)
        newImage = Image.new('RGB', (64*5, 64*5))
        
        index = 0
        for x_offset in x_offsets:
            for y_offset in y_offsets:
                newImage.paste(Images[index], (x_offset,y_offset))
                index += 1
        newImage.save(self.result_dir + "/" + str(epoch) + ".png")
      
    def save(self, epoch):
        self.save_dir = "LSGAN_saved"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        torch.save(self.G.state_dict(), os.path.join(self.save_dir, str(epoch) + "_G.pkl"))
        torch.save(self.G.state_dict(), os.path.join(self.save_dir, str(epoch) + "_D.pkl"))
        

def main():
    gan = LSGAN(Data_Train)
    gan.train()
    
if __name__ == "__main__":
    main()
