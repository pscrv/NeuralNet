using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.NetComponent
{
    public abstract class Trainer
    {
        #region protected members
        protected TrainableComponent _component;
        protected IEnumerable<TrainingVector> _trainingData;
        protected AdaptationStrategy _strategy;
        #endregion

        #region constructor
        public Trainer(TrainableComponent combiner, IEnumerable<TrainingVector> trainingdata)
        {
            _component = combiner;
            _trainingData = trainingdata;
            _strategy = new GradientDescent(1.0);
        }
        #endregion
        
        public abstract void Train();

        #region protected methods
        protected double _errorFunction(NetworkVector result, NetworkVector expected)
        {
            NetworkVector _getSquaredErrors = NetworkVector.ApplyFunctionComponentWise(result, expected, (x, y) => (x - y) * (x - y));
            return _getSquaredErrors.SumValues() / 2;
        }
        

        protected NetworkVector _getErrorGradient(TrainingVector tv)
        {
            _component.Run(tv.Input);
            NetworkVector gradient = _component.Output.Copy();
            gradient.Subtract(tv.Target);
            return gradient;
        }
        #endregion
    }



    public class OnlineTrainer2 : Trainer
    {
        #region constructors
        public OnlineTrainer2(TrainableComponent component, IEnumerable<TrainingVector> trainingdata)
            : base(component, trainingdata)
        { }
        #endregion

        #region public methods
        public override void Train()
        {
            NetworkVector errorGradient;
            NetworkMatrix weightsDelta = new NetworkMatrix(_component.NumberOfOutputs, _component.NumberOfInputs);
            NetworkVector biasDelta = new NetworkVector(_component.NumberOfOutputs);
            foreach (TrainingVector tv in _trainingData)
            {
                errorGradient = _getErrorGradient(tv);
                biasDelta = _component.BiasesGradient(errorGradient);
                weightsDelta = _component.WeightsGradient(errorGradient);
                _component.Update(
                    _strategy.BiasesUpdate(biasDelta),
                    _strategy.WeightsUpdate(weightsDelta)
                    );              
            }
        }
        #endregion
    }
    
    
    public class BatchTrainer2 : Trainer
    {
        #region constructor
        public BatchTrainer2(TrainableComponent component, IEnumerable<TrainingVector> trainingdata)
            : base (component, trainingdata)
        { }
        #endregion

        #region public methods
        public override void Train()
        {
            NetworkVector errorGradient;
            NetworkMatrix weightsDelta = new NetworkMatrix(_component.NumberOfOutputs, _component.NumberOfInputs);
            NetworkVector biasDelta = new NetworkVector(_component.NumberOfOutputs);
            foreach (TrainingVector tv in _trainingData)
            {
                errorGradient = _getErrorGradient(tv);
                biasDelta.Add(_component.BiasesGradient(errorGradient));
                weightsDelta.Add(_component.WeightsGradient(errorGradient));
            }
            _component.Update(
                _strategy.BiasesUpdate(biasDelta),
                _strategy.WeightsUpdate(weightsDelta)
                );
        }
        #endregion
        
    }
}
