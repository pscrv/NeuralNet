using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNet
{
    public class TrainableComponentBank : TrainableComponent
    {
        #region proteced attributes
        protected TrainableComponent _component;
        protected int _repetitions;
        protected NetworkVector _input;
        protected NetworkVector _output;
        #endregion

        

        #region NetComponent overrides
        public override int NumberOfInputs { get { return _input.Dimension; } }
        public override int NumberOfOutputs { get { return _output.Dimension; } }
        #endregion


        #region TrainableComponent overrides
        public override NetworkVector Biases { get { return _component.Biases; } }
        public override WeightsMatrix Weights { get { return _component.Weights; } }

        public override NetworkVector BiasesGradient(NetworkVector outputgradient)
        {
            return NetworkVector.Sum(_segment(outputgradient));
        }

        public override WeightsMatrix WeightsGradient(NetworkVector outputgradient)
        {
            WeightsMatrix weightsGradient = new WeightsMatrix(Weights.NumberOfOutputs, Weights.NumberOfInputs);
            foreach (var pair in _segmentAndPair(VectorInput, outputgradient))
            {
                weightsGradient.Add(pair.Second.OuterProduct(pair.First));
            }

            return weightsGradient;
        }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            List<NetworkVector> inputGradientParts = new List<NetworkVector>();
            foreach (NetworkVector outputGradientPart in _segment(outputgradient))
            {
                inputGradientParts.Add(Weights.DotWithWeightsPerInput(outputGradientPart));
            }

            return NetworkVector.Concatenate(inputGradientParts);
        }

        public override VectorBatch InputGradient(VectorBatch outputgradients)
        {
            if (outputgradients == null || outputgradients.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            List<VectorBatch> inputGradientParts = new List<VectorBatch>();
            foreach (VectorBatch outputGradientPart in _segment(outputgradients))
            {
                inputGradientParts.Add(Weights.LeftMultiplyBy(outputGradientPart));
            }

            return VectorBatch.Concatenate(inputGradientParts);
        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            BackPropagate(outputgradient, VectorInput);
        }

        public override void BackPropagate(NetworkVector outputgradient, NetworkVector input)
        {
            foreach (var pair in _segmentAndPair(input, outputgradient))
            {
                _component.BackPropagate(pair.Second, pair.First);
            }
        }

        public override void BackPropagate(VectorBatch outputgradient)
        {
            BackPropagate(outputgradient, BatchInput);
        }

        public override void BackPropagate(VectorBatch outputgradient, VectorBatch input)
        {
            foreach (var pair in _segmentAndPair(input, outputgradient))
            {
                _component.BackPropagate(pair.Second, pair.First);
            }
        }

        public override void Update(AdaptationStrategy strategy)
        {
            _component.Update(strategy);
        }

        protected override NetworkVector _run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("Input dimension does not match this Layer.");

            VectorInput = inputvalues;
            BatchInput = null;

            List<NetworkVector> outputParts = new List<NetworkVector>();
            foreach (NetworkVector inputPart in _segment(inputvalues))
            {
                outputParts.Add(Biases.SumWith(Weights.LeftMultiply(inputPart)));
            }

            return NetworkVector.Concatenate(outputParts);
            
        }

        protected override VectorBatch _run(VectorBatch inputbatch)
        {
            if (inputbatch.Dimension != NumberOfInputs)
                throw new ArgumentException("Input dimension does not match this Layer.");

            VectorInput = null;
            BatchInput = inputbatch;

            List<VectorBatch> outputParts = new List<VectorBatch>();
            VectorBatch result;
            foreach (VectorBatch inputPart in _segment(inputbatch))
            {
                result = Weights.TransposeAndLeftMultiplyBy(inputPart);
                result.AddVectorToEachRow(Biases);
                outputParts.Add(result);
            }

            return VectorBatch.Concatenate(outputParts);
        }
        #endregion


        #region constructors
        public TrainableComponentBank(TrainableComponent component, int repetitions)
            : base(component.NumberOfOutputs, component.NumberOfInputs)
        {
            if (repetitions < 1)
                throw new ArgumentException("Attempt to create a LayerBank with <= 0 banks.");

            _component = component;
            _repetitions = repetitions;
            _input = new NetworkVector(_repetitions * component.NumberOfInputs);
            _output = new NetworkVector(_repetitions * component.NumberOfOutputs);
        }
        #endregion


        #region protected methods
        protected List<NetworkVector> _segment(NetworkVector vectortoSegment)
        {
            return vectortoSegment.Segment(_repetitions);
        }

        protected List<VectorBatch> _segment(VectorBatch batchToSegment)
        {
            return batchToSegment.Segment(_repetitions);
        }

        protected IEnumerable<VectorPair> _segmentAndPair(NetworkVector first, NetworkVector second)
        {
            return _segment(first).Zip(_segment(second), (a, b) => new VectorPair(a, b));
        }

        protected IEnumerable<BatchPair> _segmentAndPair(VectorBatch first, VectorBatch second)
        {
            return _segment(first).Zip(_segment(second), (a, b) => new BatchPair(a, b));
        }
        #endregion

    }


    public class WeightedCombinerBank : TrainableComponentBank
    {
        public WeightedCombinerBank(WeightedCombiner combiner, int repetitions)
            : base(combiner, repetitions) { }
    }
    

    public class LayerBank : TrainableComponentBank
    {
        public LayerBank(Layer layer, int repetitions)
            : base (layer, repetitions) { }
    }
}
