from utils import flatten


class Tracker:
    def __init__(self, config):
        self.config = config

        # 'On' Condition
        self.on = self.config['track_experiment']['track']

        # Tracking configuration
        tracking_config = self.config['track_experiment']['config']
        self.mode = tracking_config['type']
        assert self.mode in ["aim", 'wandb'], "type should be either 'aim' or 'wandb'"
        self.run_id = tracking_config['run_id']
        self.context = tracking_config['context']

        # Create Tracker if 'on'
        self.tracker = self.initialize()

    @staticmethod
    def _if_on(func):
        def check_on(self, *args, **kwargs):
            return func(self, *args, **kwargs) if self.on else None
        return check_on

    @_if_on
    def initialize(self):  # Function to set up the trackers
        if self.mode == "wandb":
            import wandb
            wandb.init(config=flatten(self.config),
                       id=self.run_id,
                       project=self.context)
            return wandb

        elif self.mode == "aim":
            from aim import Run
            run = Run()
            run.name = self.run_id
            run[...] = self.config
            return run

    @_if_on
    def track(self, logs, **kwargs):  # Proxy tracking function for both cases
        if self.mode == "wandb":
            self.tracker.log(logs)
        elif self.mode == "aim":
            self.tracker.track(logs,
                               epoch=kwargs['epoch'],
                               name=None,
                               context=kwargs['context'])
