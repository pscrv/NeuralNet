using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.NetComponent
{
    public abstract class WCTrainer
    {
        #region protected members
        protected WeightedCombiner _combiner;
        protected IEnumerable<TrainingVector> _trainingData;
        protected AdaptationStrategy _strategy;
        #endregion

        #region constructor
        public WCTrainer(WeightedCombiner combiner, IEnumerable<TrainingVector> trainingdata)
        {
            _combiner = combiner;
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
            _combiner.Run(tv.Input);
            NetworkVector gradient = _combiner.Output.Copy();
            gradient.Subtract(tv.Target);
            return gradient;
        }
        #endregion
    }



    public class WCOnlineTrainer : WCTrainer
    {
        #region constructors
        public WCOnlineTrainer(WeightedCombiner combiner, IEnumerable<TrainingVector> trainingdata)
            : base(combiner, trainingdata)
        { }
        #endregion

        #region public methods
        public override void Train()
        {
            NetworkVector errorGradient;
            NetworkMatrix weightsDelta = new NetworkMatrix(_combiner.NumberOfOutputs, _combiner.NumberOfInputs);
            NetworkVector biasDelta = new NetworkVector(_combiner.NumberOfOutputs);
            foreach (TrainingVector tv in _trainingData)
            {
                errorGradient = _getErrorGradient(tv);
                biasDelta = _combiner.BiasesGradient(errorGradient);
                weightsDelta = _combiner.WeightsGradient(errorGradient);
                _combiner.Update(
                    _strategy.BiasesUpdate(biasDelta),
                    _strategy.WeightsUpdate(weightsDelta)
                    );              
            }
        }
        #endregion
    }
    
    
    public class WCBatchTrainer : WCTrainer
    {
        #region constructor
        public WCBatchTrainer(WeightedCombiner combiner, IEnumerable<TrainingVector> trainingdata)
            : base (combiner, trainingdata)
        { }
        #endregion

        #region public methods
        public override void Train()
        {
            NetworkVector errorGradient;
            NetworkMatrix weightsDelta = new NetworkMatrix(_combiner.NumberOfOutputs, _combiner.NumberOfInputs);
            NetworkVector biasDelta = new NetworkVector(_combiner.NumberOfOutputs);
            foreach (TrainingVector tv in _trainingData)
            {
                errorGradient = _getErrorGradient(tv);
                biasDelta.Add(_combiner.BiasesGradient(errorGradient));
                weightsDelta.Add(_combiner.WeightsGradient(errorGradient));
            }
            _combiner.Update(
                _strategy.BiasesUpdate(biasDelta),
                _strategy.WeightsUpdate(weightsDelta)
                );
        }
        #endregion
        
    }
}
