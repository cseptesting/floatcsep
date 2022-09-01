from gefe import models
import yaml
from datetime import datetime as dt
class NoAliasLoader(yaml.Loader):
    def ignore_aliases(self, data):
        return True

# with open('../experiment/models.yml', 'r') as r:
#     model_config = yaml.load(r, NoAliasLoader)

# team = models.Model.from_dict(model_config[0])
# team.stage()
# start = dt(2015,1,1,0,0,0)
# end = dt(2017,1,1,0,0,0)
# a = team.create_forecast(start, end, 'N10L11')
# wheel = models.Model.from_dict(model_config[1])
# wheel.stage()


with open('../experiment/tests.yml', 'r') as r:
    test_config = yaml.load(r, NoAliasLoader)
a = models.Experiment.from_yaml('../experiment/config.yml')
# a = models.Test.from_dict(test_config[0])