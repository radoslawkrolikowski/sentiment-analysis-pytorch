import torch


class EarlyStopping:
    """Early stop the training if current metric is worse than the best one for longer than
    number of wait_epochs or if metric stops changing.

    Parameters
    ----------
    wait_epochs: int, optional (default=2)
        Number of epochs to wait to improve the metric to stop the training.

    """

    def __init__(self, wait_epochs=2):

        self.wait_epochs = wait_epochs
        self.num_bad_scores = 0
        self.num_const_scores = 0
        self.best_score = None
        self.best_metric = 0


    def stop(self, metric, model, metric_type='better_decrease', delta=0.03):
        """Stop the training if metric criteria aren't met.

        Parameters
        ----------
        metric: float
            Metric used to evaluate the validation performance.
        model: pytorch model
            Pytorch model instance.
        metric_type: str, optional (default='better_decrease')
            Specify the metric type, available options: better_decrease, better_increase.
        delta: float, optional (default=0.03)
            The minimum change of a metric that is considered in stoping decision.
            Fraction of the metric.

        Returns
        -------
        Boolean
            True if training should be stoped, otherwise False.

        """
        self.delta = delta
        delta = self.delta * metric

        if self.best_score is None:
            self.best_score = metric
            self.save_model_state(metric, model)
            return False

        if abs(metric - self.best_score) < self.delta/3 * metric:
            self.num_const_scores += 1
            if self.num_const_scores >= self.wait_epochs + 1:
                print('\nTraining stoped by EarlyStopping')
                return True
            else:
                self.num_const_scores = 0

        if metric_type == 'better_decrease':
            if metric > self.best_score + delta:
                self.num_bad_scores += 1
            elif metric > self.best_score:
                self.num_bad_scores = 0
            else:
                self.best_score = metric
                self.save_model_state(metric, model)
                self.num_bad_scores = 0

        else:
            if metric < self.best_score - delta:
                self.num_bad_scores += 1
            elif metric < self.best_score:
                self.num_bad_scores = 0
            else:
                self.best_score = metric
                self.save_model_state(metric, model)
                self.num_bad_scores = 0

        if self.num_bad_scores >= self.wait_epochs:
            print('\nTraining stoped by EarlyStopping')
            return True


        return False


    def save_model_state(self, metric, model):
        """Saves the best model state.

        """
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.best_metric = metric
