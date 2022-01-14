import paddle
from tqdm import tqdm
from paddle.optimizer import AdamW
from paddle.optimizer.lr import LinearWarmup
# from transformers import WEIGHTS_NAME, AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from paddle.optimizer.lr import LRScheduler

class LinearScheduleWithWarmup(LRScheduler):
    def __init__(self,
                learning_rate,
                warmup_steps,
                max_train_steps,
                last_epoch=-1,
                verbose=False):

        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.max_train_steps = max_train_steps
        super(LinearScheduleWithWarmup, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_percent = self.last_epoch / self.warmup_steps
        else:
            warmup_percent = 1 - self.last_epoch / self.max_train_steps

        return self.learning_rate * warmup_percent


class Trainer(object):
    def __init__(self, args, model, dataloader, num_train_steps, step_callback=None):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.num_train_steps = num_train_steps
        self.step_callback = step_callback

        self.optimizer, self.scheduler = self._create_optimizer(model)
        self.wd_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm"])]

    def train(self):
        model = self.model
        max_f1 = 0
        epoch = 0
        global_step = 0
        tr_loss = 0
        total_loss = []

        model.train()

        with tqdm(total=self.num_train_steps) as pbar:
            while True:
                for step, batch in enumerate(self.dataloader):
                    # inputs = {k: v.to(self.args.device) for k, v in self._create_model_arguments(batch).items()}
                    outputs = model(word_ids=batch[0],
                                    word_segment_ids=batch[1],
                                    word_attention_mask=batch[2],
                                    entity_ids=batch[3],
                                    entity_position_ids=batch[4],
                                    entity_segment_ids=batch[5],
                                    entity_attention_mask=batch[6],
                                    labels=batch[7])

                    loss = outputs[0]

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()
                    # tr_loss += loss
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.clear_grad()
                        pbar.set_description("epoch: %d loss: %.7f f1: %.7f" % (epoch, loss, max_f1))

                        pbar.update()
                        global_step += 1
                        total_loss.append(loss.item())

                        if global_step == self.num_train_steps:
                            break
                        if self.step_callback is not None:
                            max_f1 = self.step_callback(model, global_step, max_f1, self.args.output_dir)
                            model.train()
                        
                #output_dir = self.args.output_dir

                #paddle.save(model.state_dict(), os.path.join(output_dir, 'luke.pt'))
                if global_step == self.num_train_steps:
                    import json
                    with open('total_loss.json', 'w') as f:
                        json.dump(total_loss, f)
                    break
                epoch += 1

        # logger.info("global_step = %s, average loss = %s", global_step, tr_loss.numpy() / global_step)

        return model, global_step, tr_loss / global_step

    def _create_optimizer(self, model):
        scheduler = self._create_scheduler()
        clip = paddle.nn.ClipGradByNorm(clip_norm=1.0)
        return AdamW(
            parameters=model.parameters(),
            grad_clip=clip,
            learning_rate=scheduler,
            beta1=self.args.adam_b1,
            beta2=self.args.adam_b2,
            apply_decay_param_fun=lambda x: x in self.wd_params,
            weight_decay=self.args.weight_decay
            ), scheduler

    def _create_scheduler(self):
        warmup_steps = int(self.num_train_steps * self.args.warmup_proportion)
        return LinearScheduleWithWarmup(self.args.learning_rate, warmup_steps, self.num_train_steps)
        # learing_rate, warmup_steps, start_lr, end_lr, last_epoch=- 1, verbose=False

    def _create_model_arguments(self, batch):
        return batch


