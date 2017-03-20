using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class WeightedCombiner : TrainableComponent
    {

        #region NetworkComponent overrides
        public override int NumberOfInputs { get { return Weights.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return Weights.NumberOfOutputs; } }
        
        public override NetworkVector Input { get { return _input; } protected set { _input = value; } }
        public override NetworkVector Output { get { return _output; } protected set { _output = value; } }

        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("The dimension of the input does not match this WeightedCombinger.");

            Input = inputvalues;
            Output = Biases.SumWith(Weights.LeftMultiply(inputvalues));

        }
        #endregion

        #region protected attributes
        protected NetworkVector _input;
        protected NetworkVector _output;
        #endregion

        #region public properties
        public override NetworkVector Biases { get; }
        public override NetworkMatrix Weights { get; }
        #endregion

        #region Constructors
        public WeightedCombiner(NetworkMatrix weights, NetworkVector biases)
            : base (weights.NumberOfOutputs, weights.NumberOfInputs)
        {
            if (weights == null)
                throw new ArgumentException("Attempt to make a WeightedCombineer with weights == null.");

            if (biases == null)
                throw new ArgumentException("Attempt to make a WeightedCombineer with biases == null.");

            if (biases.Dimension != weights.NumberOfOutputs)
                throw new ArgumentException("Dimension of biases must the the same of the outputs.");

            Weights = weights.Copy();
            Biases = biases.Copy();
            _input = new NetworkVector(weights.NumberOfInputs);
            _output = new NetworkVector(weights.NumberOfOutputs);
        }
        
        public WeightedCombiner(NetworkMatrix weights)
            : this(weights, new NetworkVector(weights.NumberOfOutputs)) { }
        
        
        public WeightedCombiner(WeightedCombiner combiner)
            : base (combiner.NumberOfOutputs, combiner.NumberOfInputs)
        {
            this.Biases = combiner.Biases.Copy();
            this.Weights = combiner.Weights.Copy();
        }
        #endregion


        #region TrainableComponent overrides
        public override NetworkVector BiasesGradient(NetworkVector outputgradient)
        {
            return outputgradient.Copy();
        }

        public override NetworkMatrix WeightsGradient(NetworkVector outputgradient)
        {
            return outputgradient.LeftMultiply(Input);
        }

        public override void Update(AdaptationStrategy strategy)
        {
            Biases.Add(strategy.BiasesUpdate(_biasesGradientAccumulator));
            Weights.Add(strategy.WeightsUpdate(_weightsGradientAccumulator));
            _biasesGradientAccumulator.Zero();
            _weightsGradientAccumulator.Zero();
        }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            return Weights.DotWithWeightsPerInput(outputgradient);
        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            _biasesGradientAccumulator.Add(BiasesGradient(outputgradient));
            _weightsGradientAccumulator.Add(WeightsGradient(outputgradient));
        }
        #endregion
    }
}