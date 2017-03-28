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
        protected double _costAccumulator;
        #endregion

        #region public properties
        public double Cost { get { return _costAccumulator; } }
        #endregion

        #region constructor
        public Trainer(ITrainable component, CostFunction cf, AdaptationStrategy strategy)
        {
            _component = component;
            _costFunction = cf;
            _strategy = strategy;
            _costAccumulator = 0;
        }
        #endregion

        public void Train(TrainingCollection trainingdata)
        {

            _costAccumulator = 0;
            foreach (VectorPair tv in trainingdata)
            {
                _runAndBackPropagate(tv);
                //errorGradient = _getErrorGradient(tv);
                //_component.BackPropagate(errorGradient);
            }
            _component.Update(_strategy);
        }

        public void ParallelTrain(TrainingCollection trainingdata)
        {

            _costAccumulator = 0;
            Parallel.ForEach(
                trainingdata, 
                tv =>
                {
                    _runAndBackPropagate(tv);
                    //errorGradient = _getErrorGradient(tv);
                    //_component.BackPropagate(errorGradient);
                }
                );
            _component.Update(_strategy);
        }

        #region protected methods

        protected NetworkVector _getErrorGradient(VectorPair tv)
        {
            _component.Run(tv.First);
            return _costFunction.Gradient(tv.Second, _component.Output);
        }

        protected void _runAndBackPropagate(VectorPair tv)
        {
            _component.Run(tv.First);
            _costAccumulator += _costFunction.Cost(tv.Second, _component.Output);
            _component.BackPropagate(_costFunction.Gradient(tv.Second, _component.Output));
        }
        #endregion
    }

    
}
