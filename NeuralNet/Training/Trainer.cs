using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class Trainer
    {
        #region protected fields
        protected ITrainable _component;
        protected AdaptationStrategy _strategy;
        protected CostFunction _costFunction;
        #endregion

        #region constructor
        public Trainer(ITrainable component, CostFunction cf, AdaptationStrategy strategy)
        {
            _component = component;
            _costFunction = cf;
            _strategy = strategy;
        }
        #endregion

        public void Train(TrainingCollection trainingdata)
        {
            NetworkVector errorGradient;
            foreach (VectorPair tv in trainingdata)
            {
                errorGradient = _getErrorGradient(tv);
                _component.BackPropagate(errorGradient);
            }
            _component.Update(_strategy);
        }
        
        #region protected methods

        protected NetworkVector _getErrorGradient(VectorPair tv)
        {
            _component.Run(tv.First);
            NetworkVector output = _component.Output.Copy();
            return _costFunction.Gradient(tv.Second, output);
        }
        #endregion
    }

    
}
