using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NeuralNet2;

namespace NeuralNet
{
    public abstract class AdaptationStrategy
    {
        public abstract WeightsMatrix WeightsUpdate(WeightsMatrix gradient);
        public abstract NetworkVector BiasesUpdate(NetworkVector gradient);

        public abstract NeuralNet2.WeightsMatrix WeightsUpdate(NeuralNet2.WeightsMatrix gradient);
        public abstract NeuralNet2.BiasesVector BiasesUpdate(NeuralNet2.BiasesVector gradient);
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
            result.Scale(-_stepSize / _batchSize);
            return result;
        }

        public override WeightsMatrix WeightsUpdate(WeightsMatrix gradient)
        {
            WeightsMatrix result = gradient.Copy();
            result.Scale(-_stepSize / _batchSize);
            return result;
        }

        public override BiasesVector BiasesUpdate(NeuralNet2.BiasesVector gradient)
        {
            Matrix delta = gradient.Scale(-_stepSize / _batchSize);
            
            return new BiasesVector( delta );
        }

        public override NeuralNet2.WeightsMatrix WeightsUpdate(NeuralNet2.WeightsMatrix gradient)
        {
            Matrix delta = gradient.Scale(-_stepSize / _batchSize);

            return new NeuralNet2.WeightsMatrix( delta );
        }

        #endregion
    }
}
