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
        public override NetworkVector Input { get { return _input; } set { _input = value; } }
        public override NetworkVector Output { get { return _output; } protected set { _output = value; } }

        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("Input dimension does not match this Layer.");

            Input = inputvalues;

            List<NetworkVector> outputParts = new List<NetworkVector>();
            foreach (NetworkVector inputPart in _segment(_input))
            {
                outputParts.Add(Biases.SumWith(Weights.LeftMultiply(inputPart)));
            }
            Output = NetworkVector.Concatenate(outputParts);
        }
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
            foreach (var pair in _segmentAndPair(_input, outputgradient))
            {
                weightsGradient.Add(pair.Second.LeftMultiply(pair.First));
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

        public override void BackPropagate(NetworkVector outputgradient)
        {
            foreach (var pair in _segmentAndPair(_input, outputgradient))
            {
                _component.Input = pair.First;
                _component.BackPropagate(pair.Second);
            }
        }

        public override void Update(AdaptationStrategy strategy)
        {
            _component.Update(strategy);
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

        protected IEnumerable<VectorPair> _segmentAndPair(NetworkVector first, NetworkVector second)
        {
            return _segment(first).Zip(_segment(second), (a, b) => new VectorPair(a, b));
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
