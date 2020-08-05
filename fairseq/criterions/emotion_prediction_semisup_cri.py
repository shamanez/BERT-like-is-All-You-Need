# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
import numpy as np

import csv

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


@register_criterion('emotion_prediction_semisup_cri') #This help to find the loss function acording to the task
class EmotionPredictionSemisupCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target,regression_target_mos,\
        binary_target_iemocap,softmax_target_meld,eval_metric,save_pred):
        super().__init__(task)

     
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

        self.regression_target_mos = regression_target_mos
        self.binary_target_iemocap=binary_target_iemocap
        self.softmax_target_meld=softmax_target_meld
        self.eval_metric=eval_metric

        self.save_pred=save_pred

        if self.save_pred is not None:

        
            final_header_label = ['True','Predicted']

            hap_file=open('/hpc/gsir059/INTERSPEECH/MOSI-SEMI/fairseq-wav-rob-tex-semi-sup-emocap-INT-FINAL/hap.csv', 'wt', newline ='')
            sad_file=open('/hpc/gsir059/INTERSPEECH/MOSI-SEMI/fairseq-wav-rob-tex-semi-sup-emocap-INT-FINAL/sad.csv', 'wt', newline ='')
            ang_file=open('/hpc/gsir059/INTERSPEECH/MOSI-SEMI/fairseq-wav-rob-tex-semi-sup-emocap-INT-FINAL/ang.csv', 'wt', newline ='')
            neu_file=open('/hpc/gsir059/INTERSPEECH/MOSI-SEMI/fairseq-wav-rob-tex-semi-sup-emocap-INT-FINAL/neu.csv', 'wt', newline ='')

            self.hap_file = csv.writer(hap_file, delimiter=',')
            self.sad_file = csv.writer(sad_file, delimiter=',')
            self.ang_file = csv.writer(ang_file, delimiter=',')
            self.neu_file = csv.writer(neu_file, delimiter=',')


            self.hap_file.writerow(i for i in final_header_label)
            self.sad_file.writerow(i for i in final_header_label)
            self.ang_file.writerow(i for i in final_header_label)
            self.neu_file.writerow(i for i in final_header_label)






    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        parser.add_argument('--classification-head-name',
                            default='emotion_classification_head',
                            help='name of the classification head to use')


    def forward(self, model, sample, update_num,reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training

        """
     
        assert hasattr(model, 'classification_heads') and \
            'emotion_classification_head' in model.classification_heads, \
            "model must provide emotion_classification_head for --criterion=emotion_prediction"

        


        logits_dict= model(
            sample,
            features_only=True,
            classification_head_name='emotion_classification_head',
        )


        

        if logits_dict['sup_split']=='train':
            is_training=True
        else:
            is_training=False

      
        if self.regression_target_mos:

            logits=logits_dict['sup_logits']
            targets=logits_dict['sup_targets'].view(-1)

            mixup_logits=logits_dict['mixup_logits']  #we might need to average them out
            mixup_targets=logits_dict['mixup_targets']

            ori_unsup=logits_dict['ori_uda_logits']
            aug_unsup=logits_dict['aug_uda_logits']

            contrastive_logits=logits_dict['simCLR_soft_logits']
            contrastive_soft_labels=logits_dict['simCLR_soft_labels']

  
            sample_size = targets.numel()

            logits = logits.squeeze().float()
            targets = targets.float()

            logits_ori = ori_unsup
            logits_aug = aug_unsup

        
            loss_sup = F.l1_loss(
                logits,
                targets,
                reduction='sum',
            )

            loss_uda = F.l1_loss(
                logits_ori,
                logits_aug,
                reduction='sum',
            )

            loss_mixup = F.l1_loss(
                mixup_logits,
                mixup_targets,
                reduction='sum',
            )

            contrastive_soft_labels = contrastive_soft_labels.long().cuda()

         

        

            loss_contrastive = F.nll_loss(
                F.log_softmax(contrastive_logits, dim=-1, dtype=torch.float32),
                contrastive_soft_labels,
                reduction='sum',
            )


            if is_training:
                loss=loss_sup#+loss_uda+loss_mixup+loss_contrastive/4
                #loss=loss_uda+loss_mixup/2#+loss_contrastive/4  #no_sup loss
            else:
                loss=loss_sup



    

            test_preds_a7 = torch.clamp(logits, min=-3., max=3.)
            pred=torch.round(test_preds_a7)

            test_truth_a7 = torch.clamp(targets, min=-3., max=3.)
            truth=torch.round(test_truth_a7)

            ncorrect=(pred == truth).sum().item()

            if self.prediction_h is not None:
              
                for i, (id, pred) in enumerate(zip(sample['id'].tolist(), pred.tolist())):
                    if targets is not None:
                        label = targets[i].item()
                        print('{}\t{}\t{}'.format(id, pred, label), file=self.prediction_h)
                    else:
                        print('{}\t{}'.format(id, pred), file=self.prediction_h)

            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'loss_sup': utils.item(loss_sup.data) if reduce else loss_sup.data,
                'loss_uda': utils.item(loss_uda.data) if reduce else loss_uda.data,
                'loss_mm': utils.item(loss_mixup.data) if reduce else loss_mixup.data,
                'loss_con': utils.item(loss_contrastive.data) if reduce else loss_contrastive.data,
                'ntokens': sample_size,#sample['ntokens'],
                'nsentences': sample_size,
                'sample_size': sample_size,
                'ncorrect':ncorrect
            }

         
            # pred_real_i='pred_mos_real'  #needed only when calculating the F1 score
            # truth_real_i='truth_mos_real'

            # logging_output.update({truth_real_i : targets.view(-1).cpu().detach().numpy()})
            # logging_output.update({pred_real_i : logits.view(-1).cpu().detach().numpy()})


            if self.eval_metric:  # For the binary

                test_preds_np = logits.view(-1).cpu().detach().numpy()
                test_truth_np= targets.view(-1).cpu().detach().numpy()

                exclude_zero=True

                #This gives a problem when running with with batch size of one and that batch consist of a '0' as the truth
                non_zeros = np.array([i for i, e in enumerate(test_truth_np) if e != 0 or (not exclude_zero)])


                test_preds_a7_np = np.clip(test_preds_np, a_min=-3., a_max=3.)
                test_truth_a7_np = np.clip(test_truth_np, a_min=-3., a_max=3.)



                binary_truth = (test_truth_a7_np[non_zeros] > 0)
                binary_preds = (test_preds_a7_np[non_zeros] > 0)

                ncorrect_binary=(binary_preds == binary_truth).sum().item()

                pred_i='pred_mos'
                truth_i='truth_mos'



                logging_output.update(
                ncorrect_binary=ncorrect_binary)

                logging_output.update({truth_i : binary_truth})
                logging_output.update({pred_i : binary_preds})





        elif self.binary_target_iemocap:

         
            sup_target=logits_dict['sup_targets']
            sup_logits=logits_dict['sup_logits']
            sample_size=sup_target.shape[0]

            targets = sup_target.long()
            logits=sup_logits
            ###########################################
            ##only for iemocap#########################
            logits=sup_logits.view(-1, 2)
            targets=torch.nn.functional.one_hot(targets, 4)
            targets= targets.view(-1)
            ############################################

  
            loss_pure_sup = F.nll_loss(
                F.log_softmax(logits, dim=-1, dtype=torch.float32),
                targets,
                reduction='sum',
            )



            #### only for IEMOCAP#####
            loss_pure_sup=loss_pure_sup/4 # because from one sample we make 4 losses


            loss_pure_sup=loss_pure_sup
           
            preds = logits.max(dim=1)[1]

            ncorrect=(preds == targets).sum().item()



            if is_training:
            ############## for the mixup##########

                # loss_sup_init =  F.nll_loss(
                # F.log_softmax(logits, dim=-1, dtype=torch.float32),
                # targets,
                # reduction='none')

           
                # total_steps=6320
                # tsa_thresh = self.get_tsa_thresh(update_num, total_steps) #we use linear
  
                # larger_than_threshold = torch.exp(-loss_sup_init) > tsa_thresh
                # loss_mask = torch.ones_like(targets, dtype=torch.float32)* (1 - larger_than_threshold.type(torch.float32))
                # sup_loss_masked = torch.sum(loss_sup_init * loss_mask, dim=-1) 
                # sup_sample_masked=torch.max(torch.sum(loss_mask, dim=-1))

        
                # if sup_sample_masked>0:
                #     sup_loss_masked=sup_loss_masked/sup_sample_masked

                
                
                # ram_v=self.linear_rampup(update_num) #for mixmatch

        
                # mixup_logits=logits_dict['mixup_logits']  #we might need to average them out
                # mixup_targets=logits_dict['mixup_targets']


                # mixup_logits_sup=mixup_logits[0].unsqueeze(0)
                # mixup_targets_sup=mixup_targets[0].unsqueeze(0)

                # mixup_logits_unsup=mixup_logits[1:,:].unsqueeze(0)
                # mixup_targets_unsup=mixup_targets[1:,:].unsqueeze(0)

      
                # ####### only for IEMOCAP####################
                # mixup_logits_sup=mixup_logits_sup.view(-1, 2)
                # mixup_targets_sup=mixup_targets_sup.view(-1,2)

                # mixup_logits_unsup=mixup_logits_unsup.view(-1, 2)
                # mixup_targets_unsup=mixup_targets_unsup.view(-1,2)
                # ####################################################


            
                # Lx = -torch.mean(torch.sum(F.log_softmax(mixup_logits_sup, dim=1) * mixup_targets_sup, dim=1)) #for the batchsize of T be careful
                # Lu = torch.mean((torch.softmax(mixup_logits_unsup, dim=1) - mixup_targets_unsup)**2)

                # loss=sup_loss_masked + Lx + ram_v*10*Lu
                loss=loss_pure_sup

    
            else:

                loss=loss_pure_sup




            # sample_size=sample_size*4


            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'ntokens': sample_size,#sample['ntokens'], #sample size only for the trainign phase of the binary iemocap
                'nsentences': sample_size, 
                'sample_size': sample_size,
                'ncorrect':ncorrect
            }

            ##################### This is to evaluate the binary accuracy for each emotion ##################
            if self.eval_metric:
  
  
                emos = ["Neutral", "Sad", "Angry", "Happy"]

                if self.save_pred:
            
                    csv_f =[ self.neu_file, self.sad_file, self.ang_file, self.hap_file]

                test_preds = logits.view(-1, 4, 2).cpu().detach().numpy()
                test_truth = targets.view(-1, 4).cpu().detach().numpy()

                for emo_ind in range(4):
                    
                    #print(f"{emos[emo_ind]}: ")
          
                    test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
                    test_truth_i = test_truth[:,emo_ind]

                    if self.save_pred:

                        row=[test_truth_i[0],test_preds_i[0]]

                        csv_write=csv_f[emo_ind]
                        csv_write.writerow(row)

                    

                    f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
                    acc = accuracy_score(test_truth_i, test_preds_i)


                    ncorrect_i=(test_preds_i == test_truth_i).sum().item()

                    name_i='ncorrect'+"_"+emos[emo_ind]

                    pred_i='pred_'+emos[emo_ind]
                    truth_i='truth_'+emos[emo_ind]

               
                    logging_output.update({name_i : ncorrect_i})
                    logging_output.update({truth_i : test_truth_i})
                    logging_output.update({pred_i : test_preds_i})

                    # tp = (test_truth_i * test_preds_i).sum()   #.to(torch.float32)
                    # tn = ((1 - test_truth_i) * (1 - test_preds_i))#.sum().to(torch.float32)
                    # fp = ((1 - test_truth_i) * test_preds_i).sum()#.to(torch.float32)
                    # fn = (test_truth_i * (1 - test_preds_i)).sum()#.to(torch.float32)

      

    

        elif self.softmax_target_meld:

            targets = targets.long()

            loss = F.nll_loss(
                F.log_softmax(logits, dim=-1, dtype=torch.float32),
                targets,
                reduction='sum',
            )

            preds = logits.max(dim=1)[1]
            ncorrect=(preds == targets).sum().item()
 

            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'ntokens': sample_size,#sample['ntokens'],
                'nsentences': sample_size,
                'sample_size': sample_size,
                'ncorrect':ncorrect
            }

            if self.args.eval_metric:

                preds = logits.max(dim=1)[1]
                ncorrect=(preds == targets).sum().item()

                pred_i='pred_meld'
                truth_i='truth_meld'

                logging_output.update(
                ncorrect=ncorrect)

                logging_output.update({truth_i : targets})
                logging_output.update({pred_i : preds})



          
     
        elif self.regression_target:
            
            logits = logits.squeeze().float()
            targets = targets.float()

           
            loss = F.mse_loss(
                logits,
                targets,
                reduction='sum',
            )

            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'ntokens': sample_size,#sample['ntokens'],
                'nsentences': sample_size,
                'sample_size': sample_size,
            }

       
    
        return loss, sample_size, logging_output

    # @staticmethod        #accuracy is here
    # def aggregate_logging_outputs(logging_outputs):
    #     """Aggregate logging outputs from data parallel training."""
    #     loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
    #     ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
    #     nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
    #     sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

    #     nsentences_BA=nsentences/4

    #     agg_output = {
    #         'loss': loss_sum / sample_size / math.log(2),
    #         'ntokens': ntokens,
    #         'nsentences': nsentences,
    #         'sample_size': sample_size,
    #     }

    #     if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
    #         ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
    #         agg_output.update(accuracy=ncorrect/nsentences)


    #     if 'ncorrect_Neutral' in logging_outputs[0]:
    #         ncorrect_Neutral = sum(log.get('ncorrect_Neutral', 0) for log in logging_outputs)
    #         agg_output.update(accuracy_neutral=ncorrect_Neutral/nsentences_BA)

    #         ncorrect_Sad = sum(log.get('ncorrect_Sad', 0) for log in logging_outputs)
    #         agg_output.update(accuracy_sad=ncorrect_Sad/nsentences_BA)


    #         ncorrect_Angry = sum(log.get('ncorrect_Angry', 0) for log in logging_outputs)
    #         agg_output.update(accuracy_angry=ncorrect_Angry/nsentences_BA)


    #         ncorrect_Happy = sum(log.get('ncorrect_Happy', 0) for log in logging_outputs)
    #         agg_output.update(accuracy_happy=ncorrect_Happy/nsentences_BA)


    #     #Make the batchzize one ither wuse this scikit learn thing will give wring results
    #     if 'pred_Neutral' in logging_outputs[0]:
    #         pred_Neutral= np.asarray([log.get('pred_Neutral', 0) for log in logging_outputs])
    #         truth_Neutral= np.asarray([log.get('truth_Neutral', 0) for log in logging_outputs])

    #         f1_neutral = f1_score(truth_Neutral, pred_Neutral, average='weighted')
    #         acc_neutral = accuracy_score(truth_Neutral, pred_Neutral)
    #         agg_output.update(accuracy_neu=acc_neutral)
    #         agg_output.update(f1_neu=f1_neutral)


    #         pred_Sad= np.asarray([log.get('pred_Sad', 0) for log in logging_outputs])
    #         truth_Sad= np.asarray([log.get('truth_Sad', 0) for log in logging_outputs])

            
    #         f1_sad = f1_score(truth_Sad, pred_Sad, average='weighted')
    #         acc_sad = accuracy_score(truth_Sad, pred_Sad)
    #         agg_output.update(acc_sad=acc_sad)
    #         agg_output.update(f1_sad=f1_sad)



    #         pred_Angry= np.asarray([log.get('pred_Angry', 0) for log in logging_outputs])
    #         truth_Angry= np.asarray([log.get('truth_Angry', 0) for log in logging_outputs])

            
    #         f1_angry = f1_score(truth_Angry, pred_Angry, average='weighted')
    #         acc_angry = accuracy_score(truth_Angry, pred_Angry)
    #         agg_output.update(accuracy_ang=acc_angry)
    #         agg_output.update(f1_and=f1_angry)


    #         pred_Happy= np.asarray([log.get('pred_Happy', 0) for log in logging_outputs])
    #         truth_Happy= np.asarray([log.get('truth_Happy', 0) for log in logging_outputs])  

    #         f1_happy = f1_score(truth_Happy, pred_Happy, average='weighted')
    #         acc_happy = accuracy_score(truth_Happy, pred_Happy)

    #         agg_output.update(accuracy_hap=acc_happy)
    #         agg_output.update(f1_hap=f1_happy)

            
    #     if 'pred_mos' in logging_outputs[0]:
    #         pred_mos= np.asarray([log.get('pred_mos', 0) for log in logging_outputs])
    #         truth_mos= np.asarray([log.get('truth_mos', 0) for log in logging_outputs])

    #         f1_mos = f1_score(truth_mos, pred_mos, average='weighted')
    #         acc_mos = accuracy_score(truth_mos, pred_mos)
    
    #         agg_output.update(accuracy_mos_binary=acc_mos)
    #         agg_output.update(f1_mos_binary=f1_mos)

    #     if 'pred_mos_real' in logging_outputs[0]:
    #         pred_mos_real= np.array([log.get('pred_mos_real', 0) for log in logging_outputs]).flatten()
    #         truth_mos_real= np.array([log.get('truth_mos_real', 0) for log in logging_outputs]).flatten()


    #         corr = np.corrcoef(pred_mos_real, truth_mos_real)[0][1]
     
    #         agg_output.update(corre=corr)
   


    #     if sample_size != ntokens:
    #         agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
    #     return agg_output


    def linear_rampup(self,current):


        if (current >=100):
            return 1.0
        else:

            current = np.clip(current / 100, 0.0, 1.0)
            return float(current)
       
    def get_tsa_thresh(self,global_step, num_train_steps):

        start = 1. / 4

        end=1

        training_progress = torch.tensor(float(global_step) / float(num_train_steps))
        threshold = training_progress
        # if schedule == 'linear_schedule':
        #     threshold = training_progress
        # elif schedule == 'exp_schedule':
        #     scale = 5
        #     threshold = torch.exp((training_progress - 1) * scale)
        # elif schedule == 'log_schedule':
        #     scale = 5
        #     threshold = 1 - torch.exp((-training_progress) * scale)
        output = threshold * (end - start) + start

        return output.cuda()#to('cuda')

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:

    
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

   

      
        metrics.log_scalar('loss', loss_sum / sample_size , sample_size, round=3)

       
        
        
        

        if len(logging_outputs) > 0 and 'loss_sup' in logging_outputs[0]:
            loss_sum_sup = sum(log.get('loss_sup', 0) for log in logging_outputs)
            metrics.log_scalar('loss_sup', loss_sum_sup / sample_size , sample_size, round=3)
            
        if len(logging_outputs) > 0 and 'loss_uda' in logging_outputs[0]:
            loss_sum_uda = sum(log.get('loss_uda', 0) for log in logging_outputs)
            metrics.log_scalar('loss_uda', loss_sum_uda/ sample_size , sample_size, round=3)

        if len(logging_outputs) > 0 and 'loss_mm' in logging_outputs[0]:
            loss_sum_mm = sum(log.get('loss_mm', 0) for log in logging_outputs)
            metrics.log_scalar('loss_mm', loss_sum_mm / sample_size , sample_size, round=3)
        
        if len(logging_outputs) > 0 and 'loss_con' in logging_outputs[0]:
            loss_sum_con= sum(log.get('loss_con', 0) for log in logging_outputs)
            metrics.log_scalar('loss_con', loss_sum_con / sample_size , sample_size, round=3)



        
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=3)
         

        if 'pred_mos' in logging_outputs[0]:
            pred_mos= np.asarray([log.get('pred_mos', 0) for log in logging_outputs])
            truth_mos= np.asarray([log.get('truth_mos', 0) for log in logging_outputs])

            f1_mos = f1_score(truth_mos, pred_mos, average='weighted')
            acc_mos = accuracy_score(truth_mos, pred_mos)

            ncorrect_ba = sum(log.get('ncorrect_binary', 0) for log in logging_outputs)
    
            metrics.log_scalar('binary-accuracy', 100*acc_mos, nsentences, round=3)
            metrics.log_scalar('ba-accuracy', 100.0 * ncorrect_ba / nsentences, nsentences, round=3)




        # if 'ncorrect_Neutral' in logging_outputs[0]:
        #     ncorrect_Neutral = sum(log.get('ncorrect_Neutral', 0) for log in logging_outputs)
        #     metrics.log_scalar('accuracy_neutral', 100.0 * ncorrect_Neutral / (nsentences/4), nsentences/4, round=3)
         
        #     ncorrect_Sad = sum(log.get('ncorrect_Sad', 0) for log in logging_outputs)
        #     metrics.log_scalar('accuracy_sad', 100.0 * ncorrect_Sad / (nsentences/4), nsentences/4, round=3)
            
        #     ncorrect_Angry = sum(log.get('ncorrect_Angry', 0) for log in logging_outputs)
        #     metrics.log_scalar('accuracy_angry', 100.0 * ncorrect_Angry / (nsentences/4), nsentences/4, round=3)

        #     ncorrect_Happy = sum(log.get('ncorrect_Happy', 0) for log in logging_outputs)
        #     metrics.log_scalar('accuracy_happy', 100.0 * ncorrect_Happy / (nsentences/4), nsentences/4, round=3)

        if 'ncorrect_Neutral' in logging_outputs[0]:
            ncorrect_Neutral = sum(log.get('ncorrect_Neutral', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy_neutral', 100.0 * ncorrect_Neutral / (nsentences), nsentences, round=3)
         
            ncorrect_Sad = sum(log.get('ncorrect_Sad', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy_sad', 100.0 * ncorrect_Sad / (nsentences), nsentences, round=3)
            
            ncorrect_Angry = sum(log.get('ncorrect_Angry', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy_angry', 100.0 * ncorrect_Angry / (nsentences), nsentences, round=3)

            ncorrect_Happy = sum(log.get('ncorrect_Happy', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy_happy', 100.0 * ncorrect_Happy / (nsentences), nsentences, round=3)
           

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True





