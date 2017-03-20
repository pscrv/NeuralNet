using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class ChainTrainer
    {
        #region protected members
        protected NetComponentChain _chain;
        protected IEnumerable<VectorPair> _trainingData;
       // protected AdaptationStrategy _strategy;  // will need one for each trainable component
        #endregion

        #region constructor
        public ChainTrainer(NetComponentChain chain, IEnumerable<VectorPair> trainingdata)
        {
            _chain = chain;
            _trainingData = trainingdata;
            //_strategy = new GradientDescent(1.0);
        }
        #endregion

        public abstract void Train();

        #region protected methods
        // this will need to be injectable
        protected double _errorFunction(NetworkVector result, NetworkVector expected)
        {
            NetworkVector _getSquaredErrors = NetworkVector.ApplyFunctionComponentWise(result, expected, (x, y) => (x - y) * (x - y));
            return _getSquaredErrors.SumValues() / 2;
        }


        protected NetworkVector _getErrorGradient(VectorPair tv)
        {
            _chain.Run(tv.First);
            NetworkVector gradient = _chain.Output.Copy();
            gradient.Subtract(tv.Second);
            return gradient;
        }
        #endregion
    }
}
