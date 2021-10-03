import time
import torch
import torch.nn.functional as F
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import compute_eer
from utils import AverageMeter, ProgressMeter, accuracy
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
plt.switch_backend('agg')
logger = logging.getLogger(__name__)


def train(cfg, model, optimizer, train_loader, val_loader, criterion, architect, epoch, writer_dict, lr_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    alpha_entropies = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(
        len(train_loader), batch_time, data_time, losses, top1, top5, alpha_entropies,
        prefix="Epoch: [{}]".format(epoch), logger=logger)
    writer = writer_dict['writer']

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        if lr_scheduler:
            current_lr = lr_scheduler.set_lr(optimizer, global_steps, epoch)
        else:
            current_lr = cfg.TRAIN.LR

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        input_search, target_search = next(iter(val_loader))
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)

        # step architecture
        architect.step(input_search, target_search)

        alpha_entropy = architect.model.compute_arch_entropy()
        alpha_entropies.update(alpha_entropy.mean(), input.size(0))

        # compute output
        output = model(input)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write to logger
        writer.add_scalar('lr', current_lr, global_steps)
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer.add_scalar('arch_entropy', alpha_entropies.val, global_steps)

        writer_dict['train_global_steps'] = global_steps + 1

        # log acc for cross entropy loss
        writer.add_scalar('train_acc1', top1.val, global_steps)
        writer.add_scalar('train_acc5', top5.val, global_steps)

        if i % cfg.PRINT_FREQ == 0:
            progress.print(i)

def train_ffn(cfg, model_b, model_c,model_s, ffn, optimizer, train_loader, criterion, epoch, writer_dict, lr_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader), batch_time, data_time, losses, top1, top5, prefix="Epoch: [{}]".format(epoch), logger=logger)

    writer = writer_dict['writer']
    
    ffn = ffn.train()
    end = time.time()

    for i, (input1, target1,input2,target2,input3,target3) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']
	
        if i > len(train_loader):
            break

        if lr_scheduler:
            current_lr = lr_scheduler.get_lr()
        else:
            current_lr = cfg.TRAIN.LR

        # measure data loading time
        data_time.update(time.time() - end)
        input1 = input1.cuda(non_blocking=True)
        target1 = target1.cuda(non_blocking=True)

        input2 = input2.cuda(non_blocking=True)
        target2 = target2.cuda(non_blocking=True)

        input3 = input3.cuda(non_blocking=True)
        target3 = target3.cuda(non_blocking=True)
        # compute output
        output1 = model_b(input1)
        output2 = model_c(input2)
        output3 = model_s(input3)
                
        inter_output = torch.cat((output1,output2,output3),1)
        output = ffn(inter_output)

        target = target1
        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input1.size(0))
        top5.update(acc5[0], input1.size(0))
        losses.update(loss.item(), input1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write to logger
        writer.add_scalar('lr', current_lr, global_steps)
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

        # log acc for cross entropy loss
        writer.add_scalar('train_acc1', top1.val, global_steps)
        writer.add_scalar('train_acc5', top5.val, global_steps)

        if i % cfg.PRINT_FREQ == 0:
            progress.print(i)

def test_identificate(cfg, model_b, model_c,model_s, ffn, test_loader, criterion, out="out"):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(test_loader), batch_time, losses, top1, top5, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model_b.eval()
    model_c.eval()
    model_s.eval()
    ffn.eval()
    #print(len(test_loader))
    with torch.no_grad():
        end = time.time()

        labels = []
        pred_cs= []
        target_cs = []

        for i, (input1, target1,input2,target2,input3,target3,pathmap) in enumerate(test_loader):
            input1 = input1.cuda(non_blocking=True)
            target1 = target1.cuda(non_blocking=True)

            input2 = input2.cuda(non_blocking=True)
            target2 = target2.cuda(non_blocking=True)

            input3 = input3.cuda(non_blocking=True)
            target3 = target3.cuda(non_blocking=True)
            target = target1
            # compute output
            output1 = model_b(input1)
            output2 = model_c(input2)
            output3 = model_s(input3)
            
            file1 = str(list(pathmap.keys())[0]).split("/")[-1]
            file_splits = file1.split("_")
            if len(file_splits)>2:
                file1 = file_splits[0] + "_" + file_splits[1]
                continue
            else:
                file1 = file_splits[0]
            
            inter_output = torch.cat((output1,output2,output3),1)
            
            output = ffn(inter_output)
            
            Softmax = torch.nn.Softmax(dim=1)
            output = Softmax(output)
            print(output)
            p = float(output.cpu()[0][1].item())
            print(p)
            t = int(target.cpu()[0].item())
            pred_cs.append(float(output.cpu()[0][1].item()))
            target_cs.append(int(target.cpu()[0].item()))
            #map[file1] = (p,t)
            
            with open("test_res.txt","a") as f:
                f.write("{} {} \n".format(file1,p))
        # print(labels)
    return p

def validate_identificate(cfg, model_b, model_c,model_s, ffn, test_loader, criterion, out="out"):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(test_loader), batch_time, losses, top1, top5, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model_b.eval()
    model_c.eval()
    model_s.eval()
    ffn.eval()
    print(len(test_loader))
    with torch.no_grad():
        end = time.time()

        labels = []
        pred_cs= []
        target_cs = []

        for i, (input1, target1,input2,target2,input3,target3) in enumerate(test_loader):
            input1 = input1.cuda(non_blocking=True)
            target1 = target1.cuda(non_blocking=True)

            input2 = input2.cuda(non_blocking=True)
            target2 = target2.cuda(non_blocking=True)

            input3 = input3.cuda(non_blocking=True)
            target3 = target3.cuda(non_blocking=True)
            target = target1
            # compute output
            output1 = model_b(input1)
            output2 = model_c(input2)
            output3 = model_s(input3)
                
            inter_output = torch.cat((output1,output2,output3),1)
            
            output = ffn(inter_output)

            pred_cs.append(float(output.cpu()[0][1].item()))
            target_cs.append(int(target.cpu()[0].item()))

            #output = torch.mean(output, dim=0, keepdim=True)
            #output = model.forward_classifier(output)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], input1.size(0))
            top5.update(acc5[0], input1.size(0))
            loss = criterion(output, target)

            _, pred = output.topk(1, 1, True, True)
            # print()
            losses.update(loss.item(), 1)
            labels.append([test_loader.dataset.classes1[target.cpu()[
                          0].item()], test_loader.dataset.classes1[pred.cpu()[0].item()]])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 2000 == 0:
                progress.print(i)

        target_cs = np.array(target_cs).astype(int)
        # print(pred_cs)
        # print(target_cs)
        roc = roc_auc_score(target_cs,pred_cs)

        logger.info(
            'Test Acc@1: {:.8f} Acc@5: {:.8f} roc: {:.8f}'.format(top1.avg, top5.avg, roc))
        if out != 'out':
            if not os.path.exists('results'):
                os.mkdir('results')
            with open('results/{}.txt'.format(out), 'a') as f:
                f.write('-----------------------\n')
                for l in labels:
                    f.write(str(l[0])+" "+str(l[1])+"\n")

        # print(labels)
    return top1.avg, roc



def train_from_scratch(cfg, model, optimizer, train_loader, criterion, epoch, writer_dict, lr_scheduler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader), batch_time, data_time, losses, top1, top5, prefix="Epoch: [{}]".format(epoch), logger=logger)
    writer = writer_dict['writer']

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        if lr_scheduler:
            current_lr = lr_scheduler.get_lr()
        else:
            current_lr = cfg.TRAIN.LR

        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write to logger
        writer.add_scalar('lr', current_lr, global_steps)
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

        # log acc for cross entropy loss
        writer.add_scalar('train_acc1', top1.val, global_steps)
        writer.add_scalar('train_acc5', top5.val, global_steps)

        if i % cfg.PRINT_FREQ == 0:
            progress.print(i)


def validate_verification(cfg, model, test_loader):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(test_loader), batch_time, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model.eval()
    labels, distances = [], []

    with torch.no_grad():
        end = time.time()
        for i, (input1, input2, label) in enumerate(test_loader):
            input1 = input1.cuda(non_blocking=True).squeeze(0)
            input2 = input2.cuda(non_blocking=True).squeeze(0)
            label = label.cuda(non_blocking=True)

            # compute output
            outputs1 = model(input1).mean(dim=0).unsqueeze(0)
            outputs2 = model(input2).mean(dim=0).unsqueeze(0)

            dists = F.cosine_similarity(outputs1, outputs2)
            dists = dists.data.cpu().numpy()
            distances.append(dists)
            labels.append(label.data.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 2000 == 0:
                progress.print(i)

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array(
            [subdist for dist in distances for subdist in dist])

        eer = compute_eer(distances, labels)
        logger.info('Test EER: {:.8f}'.format(np.mean(eer)))

    return eer

def validate_verification_balance(cfg, model, test_loader):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(test_loader), batch_time, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model.eval()
    labels, distances = [], []

    with torch.no_grad():
        end = time.time()
        for i, (input1, input2, label) in enumerate(test_loader):
            input1 = input1.cuda(non_blocking=True).squeeze(0)
            input2 = input2.cuda(non_blocking=True).squeeze(0)
            label = label.cuda(non_blocking=True)

            # compute output
            outputs1 = model(input1).mean(dim=0).unsqueeze(0)
            outputs2 = model(input2).mean(dim=0).unsqueeze(0)

            dists = F.cosine_similarity(outputs1, outputs2)
            dists = dists.data.cpu().numpy()
            distances.append(dists)
            labels.append(label.data.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 2000 == 0:
                progress.print(i)

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array(
            [subdist for dist in distances for subdist in dist])

        eer = compute_eer(distances, labels)
        logger.info('Test EER: {:.8f}'.format(np.mean(eer)))
        print(distances,labels)
    return eer


def validate_identification(cfg, model, test_loader,proc_num, out="out"):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader), batch_time, losses, top1, top5, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model.eval()
    print(len(test_loader))
    with torch.no_grad():
        end = time.time()

        labels = []
        for i, (input, target,pathmap) in enumerate(test_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            file1 = str(list(pathmap.keys())[0]).split("/")[-1]
            file_splits = file1.split("_")
            if len(file_splits)>2:
                file1 = file_splits[0] + "_" + file_splits[1]
                continue
            else:
                file1 = file_splits[0]

            # compute output
            output = model(input)
            output = torch.mean(output, dim=0, keepdim=True)
            output = model.forward_classifier(output)
            Softmax = torch.nn.Softmax(dim=1)
            output = Softmax(output)
            print(output)
            p = float(output.cpu()[0][1].item())
            print(p)
            t = int(target.cpu()[0].item())
            #map[file1] = (p,t)
            
            with open("val_res_{}.txt".format(proc_num),"a") as f:
                f.write("{} {} \n".format(file1,p))

        # print(labels)
    return top1.avg



def validate_identification_balance(cfg, model, test_loader, out="out"):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader), batch_time, top1, top5, prefix='Test: ', logger=logger)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        labels = []
        pred_cs= []
        target_cs = []
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda(non_blocking=True).squeeze(0)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = torch.mean(output, dim=0, keepdim=True)
            output = model.forward_classifier(output)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            
            pred_cs.append( pred.cpu()[0].item())
            target_cs.append( target.cpu()[0].item())
     

            
            labels.append([test_loader.dataset.classes[target.cpu()[
                          0].item()], test_loader.dataset.classes[pred.cpu()[0].item()]])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 2000 == 0:
                progress.print(i)
        bacc = balanced_accuracy_score(target_cs,pred_cs)
     
        print("BACC : {}".format(bacc))
            # print()


        # print(labels)
    return bacc


def validate_identification_roc(cfg, model, test_loader, out="out"):

    model.eval()

    with torch.no_grad():
       

  
        pred_cs= []
        target_cs = []
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda(non_blocking=True).squeeze(0)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            output = torch.mean(output, dim=0, keepdim=True)
            output = model.forward_classifier(output)
            # _, pred = output.topk(1, 1, True, True)
            # pred = pred.t()
        
            pred_cs.append( float(output.cpu()[0][1].item()))
            target_cs.append(int(target.cpu()[0].item()))
     
        target_cs = np.array(target_cs).astype(int)
        # print(pred_cs)
        # print(target_cs)
        roc = roc_auc_score(target_cs,pred_cs)
    
        print("roc : {}".format(roc))
            # print()


        # print(labels)
    return roc
