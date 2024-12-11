import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy.stats as stats

class trainer():
    def __init__(self, num_epochs = 100, batch_size = 10, patience = 1000, datanorm = 'quantile', fig_folder_path = './', epoch_showloss = 100):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.datanorm = datanorm
        self.fig_folder_path = fig_folder_path
        self.epoch_showloss = epoch_showloss
    
    def train_test_split(self, X, y, val_ratio = 0.2, require_dataset = False):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float)
        dataset = TensorDataset(X, y)
        val_size = int(val_ratio*len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size = len(train_dataset))
        val_loader = DataLoader(val_dataset, batch_size = len(val_dataset))

        for X_batch, y_batch in train_loader:
            X_train = X_batch
            y_train = y_batch
        for X_batch, y_batch in val_loader:
            X_val = X_batch
            y_val = y_batch
        if require_dataset:
            return X_train, y_train, X_val, y_val, train_dataset, val_dataset
        else:
            return X_train, y_train, X_val, y_val
    
    def training(self, X_train, y_train, X_val, y_val, criterion, optimizer, model):
        self.delta = criterion.delta_
        # Check if returnseparatedloss exists in criterion and get its value, default to False if not present
        self.returnseparatedloss = getattr(criterion, 'returnseparatedloss', False)
        
        # Check for GPU availability and move model to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('----------Training using: '+str(device)+ '----------')
        
        model = model.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        ymean = torch.mean(y_train) 
        ystd = torch.std(y_train)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        dataloader_train = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)
        dataloader_val = DataLoader(val_dataset, batch_size = self.batch_size, shuffle = True)
        
        train_loss_list = []
        val_loss_list = []
        PICP_train_loss_list = []
        PINAW_train_loss_list = []
        coverage_loss_list = []
        width_loss_list = []
        
        ## For early stopping ##
        best_val_loss = float('inf')
        best_model_weights = None
        patience = self.patience
        ######################
        
        for epoch in range(self.num_epochs):
            # Train a model
            model.train()
            for X_batch_train, y_batch_train in dataloader_train:
                X_batch_train = X_batch_train.to(device)
                y_batch_train = y_batch_train.to(device)
                ## Data normalization ##
                y_batch_train = (y_batch_train - ymean)/ystd
                ########################
                optimizer.zero_grad()
                
                outputs = model(X_batch_train)
                
                if self.returnseparatedloss:
                    loss, _, _ = criterion(y_batch_train, outputs[:,0], outputs[:,1])
                else:
                    loss = criterion(y_batch_train, outputs[:,0], outputs[:,1])
                loss.backward()
                optimizer.step()   
            
            # Evaluate a model
            model.eval()
            with torch.no_grad():
                # Calculate the training loss in each epoch
                outputs_train = model(X_train).detach()
                
                ## Data denormalization ##
                outputs_train = outputs_train*ystd + ymean
                if self.returnseparatedloss:
                    loss, coverage_loss, width_loss = criterion(y_train, outputs_train[:,0], outputs_train[:,1])  
                    coverage_loss_list.append(coverage_loss)
                    width_loss_list.append(width_loss)
                else:
                    loss = criterion(y_train, outputs_train[:,0], outputs_train[:,1])           
         
                train_epoch_loss = loss.item()
                train_loss_list.append(train_epoch_loss)
                # Collect PICP and PINAW for each epoch
                PICP_train_loss_list.append(self.PICP(y_train, outputs_train[:,1], outputs_train[:,0]))
                PINAW_train_loss_list.append(self.PINAW(outputs_train[:,1],outputs_train[:,0], y_train))
                # Calculate the validation loss in each epoch
                outputs_val = model(X_val).detach()
                
                ## Data denormalization ##
                outputs_val = outputs_val*ystd + ymean
                ########################                
                if self.returnseparatedloss:
                    loss, coverage_loss, width_loss = criterion(y_val, outputs_val[:,0], outputs_val[:,1])
                else:
                    loss = criterion(y_val, outputs_val[:,0], outputs_val[:,1])
                val_epoch_loss = loss.item()
                val_loss_list.append(val_epoch_loss)
                
            ## For early stopping: apply at the epoch level (evaluate from last batch) to prevent noise ##
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_train_loss = train_epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience = self.patience  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    print(f'Early stopping occurs within {epoch + 1} Epochs.') 
                    break
            #########################
                
            if (epoch + 1) % self.epoch_showloss == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')
        
        # Load the best model weights before return to user
        print(f'The best model occurs in {best_epoch + 1} Epoch with the training Loss: {best_train_loss:.4f}, the val. Loss: {best_val_loss:.4f}.')
        model.load_state_dict(best_model_weights)
        
        if self.returnseparatedloss:
            return train_loss_list, val_loss_list, model, coverage_loss_list, width_loss_list
        else:
            return train_loss_list, val_loss_list, model, PICP_train_loss_list, PINAW_train_loss_list
    
    def plotloss(self, train_loss_list, val_loss_list, PICP_plot = None, PINAW_plot = None, returnplot = False, plotname = None):
        fig, ax = plt.subplots()
        fig.set_size_inches(5,3)
        ax.plot(train_loss_list, color = 'blue', label = 'Training loss', alpha = 0.5)
        ax.plot(val_loss_list, color = 'red', label = 'Validation loss', alpha = 0.5)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training loss vs epoch')
        ax.grid()
        ax.legend()
        
        if PICP_plot != None and PINAW_plot != None:
            self.ploteachloss(PICP_plot, PINAW_plot, returnplot = returnplot, plotname = plotname)
        plt.show()
        
        if returnplot:
            fig.savefig(self.fig_folder_path+'allloss_'+plotname+".pdf",format='pdf',bbox_inches='tight',pad_inches=0,transparent=True)
        
    def ploteachloss(self, PICP_plot, PINAW_plot, returnplot = False, plotname = None):
        fig, ax = plt.subplots(ncols = 2)
        fig.set_size_inches(10,3)
        if self.returnseparatedloss:
            ax[0].plot(PICP_plot, color = 'blue', label = 'Coverage loss term', alpha = 0.5)
            ax[0].set_ylabel('Coverage loss')
            ax[1].plot(PINAW_plot, color = 'blue', label = 'Width loss term', alpha = 0.5)
            ax[1].set_ylabel('Width loss')
        else:
            ax[0].plot(PICP_plot, color = 'blue', label = 'Convergence for PICP', alpha = 0.5)
            ax[0].axhline(1-self.delta, color = 'red', linestyle = 'dashed', label = f'Expected PICP {1-self.delta}')
            ax[0].set_ylim(1 - self.delta - 0.05, 1 - self.delta + 0.05)
            ax[1].plot(PINAW_plot, color = 'blue', label = 'Convergence for PINAW', alpha = 0.5)
            ax[0].set_ylabel('PICP')
            ax[1].set_ylabel('PINAW')
        for i in range(2):
            ax[i].set_xlabel('Epoch')
            ax[i].grid()
            ax[i].legend()
        plt.show()
        
        if returnplot:
            fig.savefig(self.fig_folder_path+'eachloss_'+plotname+".pdf",format='pdf',bbox_inches='tight',pad_inches=0,transparent=True)
        
    def sort_x_toplot(self, X, y):
        sorted_indices = np.argsort(X[:,0], axis = 0).ravel()
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        return X_sorted, y_sorted
        
    def predict(self, X, model, ymean = 0, ystd = 1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        model = model.to(device)
        with torch.no_grad():
            y_pred = model(X).detach()*ystd + ymean
        return y_pred
    
    def PICP(self, y, upper, lower):
        PICP = sum((y >= lower) & (y <= upper))/y.shape[0]
        return PICP
    
    def PICPsmooth(self, y, y_pred_upper, y_pred_lower, soften_ = 160, smooth = 'sigmoid'):
        if smooth == 'sigmoid': # s = 100
            k_soft_u = torch.sigmoid(soften_ * (y_pred_upper - y))
            k_soft_l = torch.sigmoid(soften_ * (y - y_pred_lower))
            k_soft = torch.multiply(k_soft_u, k_soft_l)
        elif smooth == 'arctan': # s = 160
            k_soft = (1/torch.pi)*torch.maximum(torch.zeros(1),(torch.arctan(soften_*(y - y_pred_lower)) - torch.arctan(soften_*(y - y_pred_upper))))
        elif smooth == 'tanh': # s = 50 is good
            k_soft = (1/2)*torch.maximum(torch.zeros(1), torch.tanh(soften_*(y - y_pred_lower)) + torch.tanh(soften_*(y_pred_upper - y)))
        elif smooth == 'genlogistic':
            center = (y_pred_lower + y_pred_upper)/2
            denom = (y_pred_upper - y_pred_lower)/2
            frac = (y-center)/denom
            k_soft = 1/(1 + frac**soften_)
        else:
            raise ValueError("Input must be sigmoid, arctan, tanh, or genlogistic only")
        PICP_soft = torch.mean(k_soft)
        return PICP_soft
    
    def PINAW(self, upper, lower, ytarget, alongaxis = 0):
        PIAW = torch.mean(upper - lower, axis = alongaxis).detach()
        if ytarget is None:
            return PIAW
        else:
            if self.datanorm == 'maxmin':
                y_range = ytarget.max() - ytarget.min()
            elif self.datanorm == 'quantile':
                y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
            else:
                raise ValueError("Input must be maxmin or quantile")
            return PIAW/y_range
    
        
    def PINALW(self, upper, lower, ytarget, quantile = 0.5, alongaxis = 0):
        widtharray = upper - lower
        k = int(np.floor((1-quantile) * widtharray.shape[0]))
        PIALW = np.nanmean(torch.topk(widtharray, k, dim = alongaxis, largest=True)[0], axis = alongaxis)
        if ytarget is None:
            return PIALW
        else:
            if self.datanorm == 'maxmin':
                y_range = ytarget.max() - ytarget.min()
            elif self.datanorm == 'quantile':
                y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
            else:
                raise ValueError("Input must be maxmin or quantile")
            return PIALW/y_range
    
    def Winklerscore(self, upper, lower, y, ytarget, delta = 0.1, alongaxis = 0):
        Winkler_i = torch.abs(upper - lower) + (2/delta)*((lower - y)*(y < lower) + (y - upper)*(y > upper))
        Winkler = torch.mean(Winkler_i, axis = alongaxis)
        
        if ytarget is None:
            return Winkler
        else:
            if self.datanorm == 'maxmin':
                y_range = ytarget.max() - ytarget.min()
            elif self.datanorm == 'quantile':
                y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
            else:
                raise ValueError("Input must be maxmin or quantile")

            return Winkler/y_range      
        
    def plotresult(self, model, X_input, y_input, X_train, y_train, X_val, y_val, y_true, desiredprob = 0.9
                   , returnplot = False, plotname = None, normalized = True, numplot = 2):  
        model.eval()
#         X_input_sorted, y_input_sorted = self.sort_x_toplot(X_input, y_input)
        X_train_sorted, y_train_sorted = self.sort_x_toplot(X_train, y_train)
        X_val_sorted, y_val_sorted = self.sort_x_toplot(X_val, y_val)
        
        ## Data normalization ##
        mean_norm =  torch.mean(y_train_sorted)
        std_norm = torch.std(y_train_sorted)
#         y_train_sorted = (y_train_sorted - mean_norm)/std_norm
#         y_val_sorted = (y_val_sorted - mean_norm)/std_norm    
        ######################
        
        with torch.no_grad():
#             y_pred_all = model(X_input_sorted).detach()
            y_pred_training_data = model(X_train_sorted).detach()
            y_pred_validation_data = model(X_val_sorted).detach()
        
        if normalized:
            y_pred_training_data = y_pred_training_data*std_norm + mean_norm
            y_pred_validation_data = y_pred_validation_data*std_norm + mean_norm
            
#         x_plot_list = [X_train_sorted.ravel(), X_val_sorted.ravel(), X_input_sorted.ravel()]
#         y_data_list = [y_train_sorted.ravel(), y_val_sorted.ravel(), y_input_sorted.ravel()]
#         lower_plot = [y_pred_training_data[:,0], y_pred_validation_data[:,0], y_pred_all[:,0]]
#         upper_plot = [y_pred_training_data[:,1], y_pred_validation_data[:,1], y_pred_all[:,1]]
        
        x_plot_list = [X_train_sorted[:,0].ravel(), X_val_sorted[:,0].ravel()]
        y_data_list = [y_train_sorted.ravel(), y_val_sorted.ravel()]
        lower_plot = [y_pred_training_data[:,0], y_pred_validation_data[:,0]]
        upper_plot = [y_pred_training_data[:,1], y_pred_validation_data[:,1]]
        title_list = ['Training set', 'Test set']
        # Plot train and validation set
        if numplot == 2:
            fig, ax = plt.subplots(ncols = 2)
            fig.set_size_inches(10,5)
            for i in range(len(x_plot_list)):
                ax[i].scatter(x_plot_list[i], y_data_list[i], s = 5, alpha = 0.5, color = 'black', label = 'y')
                ax[i].plot(x_plot_list[i], lower_plot[i], color = 'green', label = 'Lower bound')
                ax[i].plot(x_plot_list[i], upper_plot[i], color = 'blue', label = 'Upper bound')
                ax[i].plot(X_input[:,0], y_true, color = 'red', linestyle = 'dashed', label = 'Noiseless y') #For a vector valued function, shows PI according to the first dim
                ax[i].legend(loc='upper right')
                ax[i].set_xlabel('x')
                ax[i].set_ylabel('y')
                ax[i].set_title(title_list[i])

                PICP = self.PICP(y_data_list[i], upper_plot[i], lower_plot[i])
                PINAW = self.PINAW(upper_plot[i], lower_plot[i], y_input) #change y into yrange
                PINALW = self.PINALW(upper_plot[i], lower_plot[i], y_input, quantile = 0.5)
                Winkler = self.Winklerscore(upper_plot[i], lower_plot[i], y_data_list[i], y_input, delta = 1 - desiredprob)

                textstr1 = '\n'.join((
                r'$PICP =%.3f$' % (PICP, ),
                r'$PINAW=%.1f$%%' % (PINAW*100, ),
                r'PINALW=%.1f%%'% (PINALW*100, ),
                r'$Winkler=%.1f$%%' % (Winkler*100, )
                ))
                props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax[i].text(0.6, 0.2, textstr1, transform=ax[i].transAxes, fontsize=10,verticalalignment='top', bbox=props1)
            plt.show()
        # Plot only validation set    
        elif numplot == 1: 
            fig, ax = plt.subplots()
            fig.set_size_inches(5,5)
            i = 1
            ax.scatter(x_plot_list[i], y_data_list[i], s = 5, alpha = 0.5, color = 'black', label = 'y')
            ax.plot(x_plot_list[i], lower_plot[i], color = 'green', label = 'Lower bound')
            ax.plot(x_plot_list[i], upper_plot[i], color = 'blue', label = 'Upper bound')
            ax.plot(X_input, y_true, color = 'red', linestyle = 'dashed', label = 'Noiseless y')
            ax.legend(loc='upper right')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(title_list[i])

            PICP = self.PICP(y_data_list[i], upper_plot[i], lower_plot[i])
            PINAW = self.PINAW(upper_plot[i], lower_plot[i], y_input) #change y into yrange
            PINALW = self.PINALW(upper_plot[i], lower_plot[i], y_input, quantile = 0.5)
            Winkler = self.Winklerscore(upper_plot[i], lower_plot[i], y_data_list[i], y_input, delta = 1 - desiredprob)

            textstr1 = '\n'.join((
            r'$PICP =%.3f$' % (PICP, ),
            r'$PINAW=%.1f$%%' % (PINAW*100, ),
            r'PINALW=%.1f%%'% (PINALW*100, ),
            r'$Winkler=%.1f$%%' % (Winkler*100, )
            ))

            # these are matplotlib.patch.Patch properties
            props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            ax.text(0.6, 0.2, textstr1, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props1)
            plt.show()
            
        else:
            ValueError("numplot should be 1 (for val dataset only) or 2 (train and val dataset)")
        if returnplot:
            fig.savefig(self.fig_folder_path+'piresult_'+plotname+".pdf",format='pdf',bbox_inches='tight',pad_inches=0,transparent=True)
            

class trainer_multistep():
    def __init__(self, num_epochs = 100, batch_size = 10, patience = 1000, datanorm = 'quantile', fig_folder_path = './', epoch_showloss = 10):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.datanorm = datanorm
        self.fig_folder_path = fig_folder_path
        self.epoch_showloss = epoch_showloss
    
    def train_test_split(self, X, y, val_ratio = 0.2, require_dataset = False):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float)
        dataset = TensorDataset(X, y)
        val_size = int(val_ratio*len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size = len(train_dataset))
        val_loader = DataLoader(val_dataset, batch_size = len(val_dataset))

        for X_batch, y_batch in train_loader:
            X_train = X_batch
            y_train = y_batch
        for X_batch, y_batch in val_loader:
            X_val = X_batch
            y_val = y_batch
        if require_dataset:
            return X_train, y_train, X_val, y_val, train_dataset, val_dataset
        else:
            return X_train, y_train, X_val, y_val
    
    def training(self, X_train, y_train, X_val, y_val, criterion, optimizer, model):
        self.delta = criterion.delta_
        # Check if returnseparatedloss exists in criterion and get its value, default to False if not present
        self.returnseparatedloss = getattr(criterion, 'returnseparatedloss', False)
        
        # Check for GPU availability and move model to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('----------Training using: '+str(device)+ '----------')
        
        model = model.to(device)
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        ymean = torch.mean(y_train) 
        ystd = torch.std(y_train)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        dataloader_train = DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)
        dataloader_val = DataLoader(val_dataset, batch_size = self.batch_size, shuffle = True)
        
        train_loss_list = []
        val_loss_list = []

        ## For early stopping ##
        best_val_loss = float('inf')
        best_model_weights = None
        patience = self.patience
        ######################
        
        for epoch in range(self.num_epochs):
            # Train a model
            model.train()
            for X_batch_train, y_batch_train in dataloader_train:
                X_batch_train = X_batch_train.to(device)
                y_batch_train = y_batch_train.to(device)
                ## Data normalization ##
                y_batch_train = (y_batch_train - ymean)/ystd
                
                optimizer.zero_grad()
                outputs = model(X_batch_train)
                loss = criterion(y_batch_train, outputs[:,0::2], outputs[:,1::2])
                loss.backward()
                optimizer.step()   
                
            # Evaluate a model
            model.eval()
            with torch.no_grad():
                # Calculate the training loss in each epoch
                outputs_train = model(X_train).detach()
                ## Data denormalization ##
                outputs_train = outputs_train*ystd + ymean
                ########################
                loss = criterion(y_train, outputs_train[:,0::2], outputs_train[:,1::2])           
         
                train_epoch_loss = loss.item()
                train_loss_list.append(train_epoch_loss)

                # Calculate the validation loss in each epoch
                outputs_val = model(X_val).detach()
                
                ## Data denormalization ##
                outputs_val = outputs_val*ystd + ymean
                ########################                
                loss = criterion(y_val, outputs_val[:,0::2], outputs_val[:,1::2])
                val_epoch_loss = loss.item()
                val_loss_list.append(val_epoch_loss)
                
            ## For early stopping: apply at the epoch level (evaluate from last batch) to prevent noise ##
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_train_loss = train_epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience = self.patience  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    print(f'Early stopping occurs within {epoch + 1} Epochs.') 
                    break
            #########################
                
            if (epoch + 1) % self.epoch_showloss == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')
        
        # Load the best model weights before return to user
        print(f'The best model occurs in {best_epoch + 1} Epoch with the training Loss: {best_train_loss:.4f}, the val. Loss: {best_val_loss:.4f}.')
        model.load_state_dict(best_model_weights)
        
        return train_loss_list, val_loss_list, model
    
    def plotloss(self, train_loss_list, val_loss_list, returnplot = False, plotname = None):
        fig, ax = plt.subplots()
        fig.set_size_inches(5,3)
        ax.plot(train_loss_list, color = 'blue', label = 'Training loss', alpha = 0.5)
        ax.plot(val_loss_list, color = 'red', label = 'Validation loss', alpha = 0.5)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training loss vs epoch')
        ax.grid()
        ax.legend()
        plt.show()
        
        if returnplot:
            fig.savefig(self.fig_folder_path+'allloss_'+plotname+".pdf",format='pdf',bbox_inches='tight',pad_inches=0,transparent=True)
        
    def predict(self, X, model, ymean = 0, ystd = 1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        model = model.to(device)
        with torch.no_grad():
            y_pred = model(X).detach()*ystd + ymean
        return y_pred
    
    def PICP(self, y, upper, lower):
        PICP = sum((y > lower) & (y < upper))/y.shape[0]
        return PICP
    
    
    def PINAW(self, upper, lower, ytarget, alongaxis = 0):
        PIAW = torch.mean(upper - lower, axis = alongaxis).detach()
        if ytarget is None:
            return PIAW
        else:
            if self.datanorm == 'maxmin':
                y_range = ytarget.max() - ytarget.min()
            elif self.datanorm == 'quantile':
                y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
            else:
                raise ValueError("Input must be maxmin or quantile")
            return PIAW/y_range
    
    def PINALW(self, upper, lower, ytarget, quantile = 0.5, alongaxis = 0):
        widtharray = upper - lower
        k = int(np.floor((1-quantile) * widtharray.shape[0]))
        PIALW = np.nanmean(torch.topk(widtharray, k, dim = alongaxis, largest=True)[0], axis = alongaxis)
        if ytarget is None:
            return PIALW
        else:
            if self.datanorm == 'maxmin':
                y_range = ytarget.max() - ytarget.min()
            elif self.datanorm == 'quantile':
                y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
            else:
                raise ValueError("Input must be maxmin or quantile")
            return PIALW/y_range

    def Winklerscore(self, upper, lower, y, ytarget, delta = 0.1, alongaxis = 0):
        Winkler_i = torch.abs(upper - lower) + (2/delta)*((lower - y)*(y < lower) + (y - upper)*(y > upper))
        Winkler = torch.mean(Winkler_i, axis = alongaxis)
        
        if ytarget is None:
            return Winkler
        else:
            if self.datanorm == 'maxmin':
                y_range = ytarget.max() - ytarget.min()
            elif self.datanorm == 'quantile':
                y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
            else:
                raise ValueError("Input must be maxmin or quantile")

            return Winkler/y_range   
    
        
