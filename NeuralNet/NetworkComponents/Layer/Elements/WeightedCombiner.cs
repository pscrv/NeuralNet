using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class WeightedCombiner : NetworkComponent
    {
        #region private attributes
        protected NetworkVector _inputs;
        protected NetworkMatrix _weights;
        protected NetworkVector _biases;
        protected NetworkVector _outputs;
        protected NetworkVector _inputGradient;
        #endregion

        #region public properties
        public override int NumberOfInputs { get { return _weights.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return _weights.NumberOfOutputs; } }
        public override NetworkVector InputGradient { get { return _inputGradient; } }
        public override NetworkVector Output { get { return _outputs; } }
        public LayerState State { get { return  new LayerState(_weights, _biases); } }
        #endregion

        #region Constructors
        public WeightedCombiner(double[,] weights, double[] biases = null)
        {
            _weights = new NetworkMatrix(weights);

            _inputs = new NetworkVector(NumberOfInputs);
            _outputs = new NetworkVector(NumberOfOutputs);
            _inputGradient = new NetworkVector(NumberOfInputs);

            if (biases != null)
            {
                if (biases.Length != NumberOfOutputs)
                    throw new ArgumentException("Length of biases must the the same as the number of neurons.");
                _biases = new NetworkVector(biases);
            }
            else
            {
                _biases = new NetworkVector(NumberOfOutputs);
            }
        }

        public WeightedCombiner(WeightedCombiner combiner)
        {
            if (combiner == null)
                throw new ArgumentException("Attempt to make a WeightedCombiner from null");

            this._biases = combiner._biases.Copy();
            this._weights = combiner._weights;
        }
        #endregion

        #region public methods
        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("The dimension of the input does not match this WeightedCombinger.");
            _inputs = inputvalues;
            _outputs = _biases.SumWith(_weights.LeftMultiply(_inputs));

        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");
            
            _inputGradient = _weights.DotWithWeightsPerInput(outputgradient);
            _biases.Subtract(outputgradient);
            _weights.Subtract(outputgradient.LeftMultiply(_inputs));
        }
        #endregion
    }

    
}