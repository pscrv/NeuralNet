using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class WeightedCombiner : TrainableComponent
    {
        #region protected attributes
        #endregion

        #region public properties
        //public NetworkVector Output { get; protected set; }
        public override NetworkVector Biases { get; }
        public override WeightsMatrix Weights { get; }
        #endregion

        #region Constructors
        public WeightedCombiner(WeightsMatrix weights, NetworkVector biases)
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
            VectorInput = new NetworkVector(weights.NumberOfInputs);
            //Output = new NetworkVector(weights.NumberOfOutputs);
        }
        
        public WeightedCombiner(WeightsMatrix weights)
            : this(weights, new NetworkVector(weights.NumberOfOutputs)) { }        
        
        public WeightedCombiner(WeightedCombiner combiner)
            : base (combiner.NumberOfOutputs, combiner.NumberOfInputs)
        {
            this.Biases = combiner.Biases.Copy();
            this.Weights = combiner.Weights.Copy();
        }
        #endregion



        #region NetworkComponent overrides
        public override int NumberOfInputs { get { return Weights.NumberOfInputs; } }
        public override int NumberOfOutputs { get { return Weights.NumberOfOutputs; } }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            return Weights.DotWithWeightsPerInput(outputgradient);
        }

        public override VectorBatch InputGradient(VectorBatch outputgradients)
        {
            if (outputgradients == null || outputgradients.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");
            
            return Weights.LeftMultiplyBy(outputgradients);
        }

        
        #endregion


        #region TrainableComponent overrides
        protected override NetworkVector _run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("The dimension of the input does not match this WeightedCombinger.");

            VectorInput = inputvalues;
            BatchInput = null;
            
            return Biases.SumWith(Weights.LeftMultiply(inputvalues));
        }

        protected override VectorBatch _run(VectorBatch inputbatch)
        {
            if (inputbatch.Dimension != NumberOfInputs)
                throw new ArgumentException("The dimension of the input does not match this WeightedCombiner.");

            VectorInput = null;
            BatchInput = inputbatch;

            VectorBatch result = Weights.TransposeAndLeftMultiplyBy(inputbatch);
            result.AddVectorToEachRow(Biases);
            return result;            
        }

        public override NetworkVector BiasesGradient(NetworkVector outputgradient)
        {
            return outputgradient.Copy();
        }

        public NetworkVector BiasesGradient(VectorBatch outputgradients)
        {
            var y = outputgradients.AsMatrix().ColumnSums();
            return new NetworkVector(y);
        }

        public override WeightsMatrix WeightsGradient(NetworkVector outputgradient)
        {
            return outputgradient.OuterProduct(VectorInput);
        }

        public WeightsMatrix WeightsGradient(NetworkVector outputgradient, NetworkVector input)
        {
            return outputgradient.OuterProduct(input);
        }

        public WeightsMatrix WeightsGradient(VectorBatch outputgradients, VectorBatch inputs)
        {
            return outputgradients.LeftMultiply(inputs);
        }

        public override void Update(AdaptationStrategy strategy)
        {
            Biases.Add(strategy.BiasesUpdate(_biasesGradientAccumulator));
            Weights.Add(strategy.WeightsUpdate(_weightsGradientAccumulator));
            _biasesGradientAccumulator.Zero();
            _weightsGradientAccumulator.Zero();
        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            BackPropagate(outputgradient, VectorInput);
        }

        public override void BackPropagate(VectorBatch outputgradients)
        {
            BackPropagate(outputgradients, BatchInput);
        }


        public override void BackPropagate(NetworkVector outputgradient, NetworkVector input)
        {
            _biasesGradientAccumulator.Add(BiasesGradient(outputgradient));
            _weightsGradientAccumulator.Add(WeightsGradient(outputgradient, input));
        }

        public override void BackPropagate(VectorBatch outputgradients, VectorBatch inputs)
        {
            _biasesGradientAccumulator.Add(BiasesGradient(outputgradients));            
            _weightsGradientAccumulator.Add(WeightsGradient(outputgradients, inputs));
        }
        #endregion
    }
}