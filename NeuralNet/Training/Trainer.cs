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
}
