using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2
{
    public abstract class TrainableComponent : Component
    {
        #region attributes
        protected WeightsMatrix _weights;
        protected BiasesVector _biases;

        private VectorBatch _input;
        #endregion


        #region constructors
        public TrainableComponent(WeightsMatrix weights, BiasesVector biases)
            : base(weights.NumberOfInputs, weights.NumberOfOutputs)
        {
            if (weights.NumberOfOutputs != biases.Dimension)
                throw new ArgumentException("Mismatched WeightsMatrix and BiasesVector.");

            _weights = weights;
            _biases = biases;
        }
        #endregion


        #region Component overrides
        protected override VectorBatch _run(VectorBatch input)
        {
            _input = input;
            return _trainingRun(_input);
        }

        protected override VectorBatch _backPropate(VectorBatch outputGradient)
        {
            VectorBatch inputGradient = _getInputGradient(outputGradient);
            _updateWeights(outputGradient);
            _updateBiases(outputGradient);
            return inputGradient;
        }
        #endregion


        #region abstract methods
        protected abstract VectorBatch _trainingRun(VectorBatch input);
        protected abstract VectorBatch _getInputGradient(VectorBatch outputGradient);
        protected abstract void _updateWeights(VectorBatch outputGradient);
        protected abstract void _updateBiases(VectorBatch outputGradient);
        #endregion


    }
}
