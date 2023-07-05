import numpy as np

import torch

def test_acc(model, data_loader, device):
    correct=0
    model.eval()
    with torch.no_grad():
        for imgs, y in data_loader:
            imgs = imgs.view(y.size(0), 784).to(device)
            y = y.to(device)
            output = model(imgs)
            ypred = output.data.max(1, keepdim=True)[1].squeeze()
            correct += ypred.eq(y).sum()
    acc = correct/len(data_loader.dataset)
    return acc




def train_model(model, train_loader_cls, train_loader_mem,
                optimizer_cls, optimizer_mem, loss_cls, loss_mem,
                epochs, save_path, device):
    
    best_loss_r = np.inf
    epoch = 0
    for epoch in range(epochs):
        loss_c = 0
        loss_r = 0
        c=0
        mem_iterator = iter(train_loader_mem)
        for  (data, labels) in train_loader_cls:     
            try:
                (code, _, imgs) = next(mem_iterator)
            except:
                mem_iterator = iter(train_loader_mem)
                (code, _, imgs) = next(mem_iterator)
                                                  
            data = data.to(device)
            code = code.to(device)
            imgs = imgs.to(device)
            labels = labels.to(device)


            optimizer_cls.zero_grad()
            optimizer_mem.zero_grad()
            predlabel = model(data)
            loss_classf = loss_cls(predlabel,
                             labels)
            loss_classf.backward()   
            optimizer_cls.step()

            optimizer_mem.zero_grad()
            optimizer_cls.zero_grad()
            predimg = model.forward_transposed(code)
            loss_recon = loss_mem(predimg, imgs)
            loss_recon.backward()
            optimizer_mem.step()

            # add the mini-batch training loss to epoch loss
            loss_c += loss_classf.item()
            loss_r += loss_recon.item()
            c+=1
        # display the epoch training loss
        print("epoch : {}/{}, loss_c = {:.6f}, loss_r = {:.6f}".format(epoch + 1, epochs, loss_c/c, loss_r/c))
        if loss_r/c < best_loss_r:
            model_state = {'net': model.state_dict(),
                           'opti_mem': optimizer_mem.state_dict(), 
                           'opti_cls': optimizer_cls.state_dict(), 
                           'loss_r': loss_r/c}
            torch.save(model_state, save_path)
            best_loss_r = loss_r/c