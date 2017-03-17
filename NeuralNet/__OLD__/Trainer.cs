using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class OnlineTrainer
    {
        #region protected members
        protected NetworkComponent _component;
        protected IEnumerable<TrainingVector> _trainingData;
        #endregion

        #region constructor
        public OnlineTrainer(NetworkComponent componenttotrain, IEnumerable<TrainingVector> trainingdata)
        {
            _component = componenttotrain;
            _trainingData = trainingdata;
        }
        #endregion

        #region public methods
        public void Train()
        {
            double error;
            NetworkVector errorgradient;
            foreach (TrainingVector tv in _trainingData)
            {
                _component.Run(tv.Input);
                error = _errorFunction(_component.Output, tv.Target);
                errorgradient = _errorGradient(_component.Output, tv.Target);
                _component.BackPropagate(errorgradient);
            }
        }
        #endregion

        #region private methods
        private double _errorFunction(NetworkVector result, NetworkVector expected)
        {
            NetworkVector squaredErrors = NetworkVector.ApplyFunctionComponentWise(result, expected, (x, y) => (x - y) * (x - y));
            return squaredErrors.SumValues() / 2;
        }

        private NetworkVector _errorGradient(NetworkVector result, NetworkVector expected)
        {
            NetworkVector gradient = result.Copy();
            gradient.Subtract(expected);
            return gradient;
        }
        #endregion
    }

   
    public class BatchTrainer
    {
        #region protected members
        protected NetworkComponent _component;
        protected IEnumerable<TrainingVector> _trainingData;
        #endregion

        #region constructor
        public BatchTrainer(NetworkComponent componenttotrain, IEnumerable<TrainingVector> trainingdata)
        {
            _component = componenttotrain;
            _trainingData = trainingdata;
        }
        #endregion

        #region public methods
        public void Train()
        {
            List<TrainingVector> results = new List<TrainingVector>();
            foreach (TrainingVector tv in _trainingData)
            {
                _component.Run(tv.Input);
                results.Add(new TrainingVector(_component.Output, tv.Target));
            }
            double error = _errorFunction(results);
            NetworkVector errorGradient = _errorGradient(results);
            _component.BackPropagate(errorGradient);
        }
        #endregion

        #region private methods
        private double _errorFunction(IEnumerable<TrainingVector> results)
        {
            double error = 0.0;
            foreach (TrainingVector result in results)
            {
                NetworkVector squaredErrors = NetworkVector.ApplyFunctionComponentWise(result.Input, result.Target, (x, y) => (x - y) * (x - y));
                error += squaredErrors.SumValues() / 2;
            }
            return error;
        }

        private NetworkVector _errorGradient(List<TrainingVector> results)
        {
            NetworkVector gradient = new NetworkVector(_component.NumberOfOutputs);
            foreach (TrainingVector result in results)
            {
                NetworkVector gradientPart = result.Input.Copy();
                gradientPart.Subtract(result.Target);
                gradient.Add(gradientPart);
            }
            return gradient;
        }
        #endregion

    }
}
