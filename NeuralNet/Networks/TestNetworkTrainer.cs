using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class NetworkTrainer
    {
        #region protected members
        protected LinearTwoLayerTestNetwork _network;
        protected IEnumerable<VectorPair> _trainingData;
        protected AdaptationStrategy _strategy;
        #endregion

        #region constructor
        public NetworkTrainer(LinearTwoLayerTestNetwork network, IEnumerable<VectorPair> trainingdata)
        {
            _network = network;
            _trainingData = trainingdata;
            _strategy = new GradientDescent(1.0, 1);
        }
        #endregion
        
        public abstract void Train();

        #region protected methods
        protected double _errorFunction(NetworkVector result, NetworkVector expected)
        {
            NetworkVector _getSquaredErrors = NetworkVector.ApplyFunctionComponentWise(result, expected, (x, y) => (x - y) * (x - y));
            return _getSquaredErrors.SumValues() / 2;
        }
        

        protected NetworkVector _getErrorGradient(VectorPair tv)
        {
            NetworkVector gradient = _network.Run(tv.First);
            gradient.Subtract(tv.Second);
            return gradient;
        }
        #endregion
    }

    

    public class OnlineNetworkTrainer : NetworkTrainer
    {
        #region constructors
        public OnlineNetworkTrainer(LinearTwoLayerTestNetwork network, IEnumerable<VectorPair> trainingdata)
            : base(network, trainingdata)
        { }
        #endregion

        #region NetworkTrainer overrides
        public override void Train()
        {
            NetworkVector errorGradient;
            foreach (VectorPair tv in _trainingData)
            { 
                errorGradient = _getErrorGradient(tv);
                _network.BackPropagate(errorGradient);
                _network.Update(_strategy);
            }
        }
        #endregion
    }


    public class BatchNetworkTrainer : NetworkTrainer
    {
        #region constructor
        public BatchNetworkTrainer(LinearTwoLayerTestNetwork network, IEnumerable<VectorPair> trainingdata)
            : base(network, trainingdata)
        { }
        #endregion

        #region public methods
        public override void Train()
        {
            NetworkVector errorGradient;
            foreach (VectorPair tv in _trainingData)
            {
                errorGradient = _getErrorGradient(tv);
                _network.BackPropagate(errorGradient);
            }

            _network.Update(_strategy);
        }
        #endregion

    }
}
