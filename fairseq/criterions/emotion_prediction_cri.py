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


@register_criterion('emotion_prediction_cri') #This help to find the loss function acording to the task
class EmotionPredictionCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target,regression_target_mos,\
        binary_target_iemocap,softmax_target_meld,eval_metric,save_predictions):
        super().__init__(task)

     
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

        self.regression_target_mos = regression_target_mos
        self.binary_target_iemocap=binary_target_iemocap
        self.softmax_target_meld=softmax_target_meld
        self.eval_metric=eval_metric

        final_header_label = ['True','Predicted']
        mos_file=open('/hpc/gsir059/INTERSPEECH/MOSI-SEMI/fairseq-wav-rob-tex-semi-sup-emocap-test-mosei/mos.csv', 'wt', newline ='')
        self.mos_file = csv.writer(mos_file, delimiter=',')
        self.mos_file.writerow(i for i in final_header_label)

        if save_predictions is not None:
            final_header_label = ['True','Predicted']
            mos_file=open('/hpc/gsir059/INTERSPEECH/MOSI-SEMI/fairseq-wav-rob-tex-semi-sup-emocap-test-mosei/mos.csv', 'wt', newline ='')
            self.mos_file = csv.writer(mos_file, delimiter=',')
            self.mos_file.writerow(i for i in final_header_label)
            #self.prediction_h = open(save_predictions, 'w')
        else:
            self.prediction_h = None






    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        parser.add_argument('--classification-head-name',
                            default='emotion_classification_head',
                            help='name of the classification head to use')
       

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training

        """
     
        assert hasattr(model, 'classification_heads') and \
            'emotion_classification_head' in model.classification_heads, \
            "model must provide emotion_classification_head for --criterion=emotion_prediction"

   
       

        logits, _ = model(
            sample['net_input'],
            features_only=True,
            classification_head_name='emotion_classification_head',
        )

    
        
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

     



      
        if self.regression_target_mos:

          
    
            logits = logits.squeeze().float()
            targets = targets.float()

        
            loss = F.l1_loss(
                logits,
                targets,
                reduction='sum',
            )

           

            test_preds_a7 = torch.clamp(logits, min=-3., max=3.)
            pred=torch.round(test_preds_a7)

            test_truth_a7 = torch.clamp(targets, min=-3., max=3.)
            truth=torch.round(test_truth_a7)

            ncorrect=(pred == truth).sum().item()


              
    

            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
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



                row=[binary_truth[0],binary_preds[0]]
                self.mos_file.writerow(row)

        

                ncorrect_binary=(binary_preds == binary_truth).sum().item()

                pred_i='pred_mos'
                truth_i='truth_mos'



                logging_output.update(
                ncorrect_binary=ncorrect_binary)

                logging_output.update({truth_i : binary_truth})
                logging_output.update({pred_i : binary_preds})





        elif self.binary_target_iemocap:

         

            targets = targets.long()
          
            targets=torch.nn.functional.one_hot(targets, 4)


            logits=logits.view(-1, 2)
            targets= targets.view(-1)

    
            

            loss = F.nll_loss(
                F.log_softmax(logits, dim=-1, dtype=torch.float32),
                targets,
                reduction='sum',
            )


        
           
            preds = logits.max(dim=1)[1]

            ncorrect=(preds == targets).sum().item()


            # sample_size=sample_size*4


            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'ntokens': sample_size*4,#sample['ntokens'], #sample size only for the trainign phase of the binary iemocap
                'nsentences': sample_size*4, 
                'sample_size': sample_size*4,
                'ncorrect':ncorrect
            }

            ##################### This is to evaluate the binary accuracy for each emotion ##################
            if self.eval_metric:
  
  
                emos = ["Neutral", "Sad", "Angry", "Happy"]

                test_preds = logits.view(-1, 4, 2).cpu().detach().numpy()
                test_truth = targets.view(-1, 4).cpu().detach().numpy()

                for emo_ind in range(4):
                    
                    #print(f"{emos[emo_ind]}: ")
          
                    test_preds_i = np.argmax(test_preds[:,emo_ind],axis=1)
                    test_truth_i = test_truth[:,emo_ind]


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



    @staticmethod
    def reduce_metrics(logging_outputs) -> None:

        
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

   

      
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
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




        if 'ncorrect_Neutral' in logging_outputs[0]:
            ncorrect_Neutral = sum(log.get('ncorrect_Neutral', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy_neutral', 100.0 * ncorrect_Neutral / (nsentences/4), nsentences/4, round=3)
         
            ncorrect_Sad = sum(log.get('ncorrect_Sad', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy_sad', 100.0 * ncorrect_Sad / (nsentences/4), nsentences/4, round=3)
            
            ncorrect_Angry = sum(log.get('ncorrect_Angry', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy_angry', 100.0 * ncorrect_Angry / (nsentences/4), nsentences/4, round=3)

            ncorrect_Happy = sum(log.get('ncorrect_Happy', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy_happy', 100.0 * ncorrect_Happy / (nsentences/4), nsentences/4, round=3)
           

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


