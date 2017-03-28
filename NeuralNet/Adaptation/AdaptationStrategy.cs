using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class AdaptationStrategy
    {
        public abstract WeightsMatrix WeightsUpdate(WeightsMatrix gradient);
        public abstract NetworkVector BiasesUpdate(NetworkVector gradient);
    }


    public class GradientDescent : AdaptationStrategy
    {
        private double _stepSize;
        private int _batchSize;

        public GradientDescent(double stepsize, int batchsize)
        {
            _stepSize = stepsize;
            _batchSize = batchsize;
        }

        public GradientDescent() : this(stepsize: 1.0, batchsize: 1) { }

        #region AdaptationStrategy overrides
        public override NetworkVector BiasesUpdate(NetworkVector gradient)
        {
            NetworkVector result = gradient.Copy();
            result.MultiplyBy(-_stepSize / _batchSize);
            return result;
        }

        public override WeightsMatrix WeightsUpdate(WeightsMatrix gradient)
        {
            WeightsMatrix result = gradient.Copy();
            result.MultiplyBy(-_stepSize / _batchSize);
            return result;
        }
        #endregion
    }
}
