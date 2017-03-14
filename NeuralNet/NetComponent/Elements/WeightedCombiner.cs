using System;
using System.Collections.Generic;

namespace NeuralNet.NetComponent
{
    public class WeightedCombiner : NetComponent
    {
        #region NetworkComponent overrides
        public override int NumberOfInputs { get { return Weights.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return Weights.NumberOfOutputs; } }
        
        public override NetworkVector Output { get; protected set; }

        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("The dimension of the input does not match this WeightedCombinger.");
            Input = inputvalues;
            Output = Biases.SumWith(Weights.LeftMultiply(Input));

        }
        #endregion

        #region public properties
        public NetworkVector Input { get; protected set; }
        public NetworkVector Biases { get; protected set; }
        public NetworkMatrix Weights { get; protected set; }
        #endregion

        #region Constructors
        public WeightedCombiner(NetworkMatrix weights, NetworkVector biases)
        {
            if (weights == null)
                throw new ArgumentException("Attempt to create a WeightedCombiner with null weights.");

            if (biases == null)
                throw new ArgumentException("Attempt to create a WeightedCombiner with null biases.");

            if (biases.Dimension != weights.NumberOfOutputs)
                throw new ArgumentException("Dimension of biases must the the same of the outputs.");

            Weights = weights.Copy();
            Biases = biases.Copy();

            Input = new NetworkVector(NumberOfInputs);
            Output = new NetworkVector(NumberOfOutputs);
        }
        
        public WeightedCombiner(NetworkMatrix weights)
            : this(weights, new NetworkVector(weights.NumberOfOutputs)) { }
        
        
        public WeightedCombiner(WeightedCombiner combiner)
        {
            if (combiner == null)
                throw new ArgumentException("Attempt to make a WeightedCombiner from null");

            this.Biases = combiner.Biases.Copy();
            this.Weights = combiner.Weights.Copy();
        }
        #endregion


        #region public methods
        public NetworkVector BiasesGradient(NetworkVector outputgradient)
        {
            return outputgradient.Copy();
        }

        public NetworkMatrix WeightsGradient(NetworkVector outputgradient)
        {
            return outputgradient.LeftMultiply(Input);
        }

        public void Update(NetworkVector biasesdelta, NetworkMatrix weightsdelta)
        {
            Biases.Add(biasesdelta);
            Weights.Add(weightsdelta);
        }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            return Weights.DotWithWeightsPerInput(outputgradient);
        }
        #endregion
    }
}