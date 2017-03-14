using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public abstract class AdaptationStrategy
    {
        public abstract NetworkMatrix WeightsUpdate(NetworkMatrix gradient);
        public abstract NetworkVector BiasesUpdate(NetworkVector gradient);
    }


    public class GradientDescent : AdaptationStrategy
    {
        private double _stepSize; 

        public GradientDescent(double stepsize)
        {
            _stepSize = stepsize;
        }

        public GradientDescent() : this(stepsize: 1.0) { }

        #region AdaptationStrategy overrides
        public override NetworkVector BiasesUpdate(NetworkVector gradient)
        {
            NetworkVector result = gradient.Copy();
            result.MultiplyBy(-_stepSize);
            return result;
        }

        public override NetworkMatrix WeightsUpdate(NetworkMatrix gradient)
        {
            NetworkMatrix result = gradient.Copy();
            result.MultiplyBy(-_stepSize);
            return result;
        }
        #endregion
    }
}
