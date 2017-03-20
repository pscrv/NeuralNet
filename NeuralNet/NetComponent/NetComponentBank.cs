using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNet
{
    public class WeightedCombinerBank : WeightedCombiner
    {
        #region proteced attributes
        protected int _repetitions;
        protected List<NetworkVector> _inputParts;
        protected new NetworkVector _input;
        protected new NetworkVector _output;
        #endregion

        #region NetComponent overrides
        public override int NumberOfInputs { get { return _input.Dimension; } }
        public override int NumberOfOutputs { get { return _output.Dimension; } }
        public override NetworkVector Input { get { return _input; } protected set { _input = value; } } 
        public override NetworkVector Output { get { return _output; } protected set { _output = value; } }

        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues.Dimension != NumberOfInputs)
                throw new ArgumentException("Input dimension does not match this Layer.");

            Input = inputvalues;
            _inputParts = inputvalues.Segment(_repetitions);

            List<NetworkVector> outputParts = new List<NetworkVector>();
            foreach (NetworkVector inputPart in _inputParts)
            {
                outputParts.Add(Biases.SumWith(Weights.LeftMultiply(inputPart)));
            }
            Output = NetworkVector.Concatenate(outputParts);
        }
        #endregion


        #region constructors
        public WeightedCombinerBank(WeightedCombiner combiner, int repetitions)
            : base(combiner)
        {
            if (repetitions < 1)
                throw new ArgumentException("Attempt to create a LayerBank with < 1 banks.");

            _repetitions = repetitions;
            _input = new NetworkVector(_repetitions * combiner.NumberOfInputs);
            _inputParts = _input.Segment(_repetitions);
            _output = new NetworkVector(_repetitions * combiner.NumberOfOutputs);
        }
        #endregion


        #region TrainableComponent overrides
        public override NetworkVector BiasesGradient(NetworkVector outputgradient)
        {
            List<NetworkVector> outputGradientParts = outputgradient.Segment(_repetitions); 
            NetworkVector biasesGradient = NetworkVector.Sum(outputGradientParts);
            return biasesGradient;
        }

        public override NetworkMatrix WeightsGradient(NetworkVector outputgradient)
        {
            List<NetworkVector> outputGradientParts = outputgradient.Segment(_repetitions);

            var partArrays =
                _inputParts.Zip(outputGradientParts, (a, b) => new { InputPart = a.ToArray(), OutputGradientPart = b.ToArray() });

            double[,] weightsGradientMatrix = new double[Weights.NumberOfOutputs, Weights.NumberOfInputs];

            foreach (var pair in partArrays)
                for (int i = 0; i < Weights.NumberOfOutputs; i++)
                    for (int j = 0; j < Weights.NumberOfInputs; j++)
                    {
                        weightsGradientMatrix[i, j] += pair.OutputGradientPart[i] * pair.InputPart[j];
                    }
            return new NetworkMatrix(weightsGradientMatrix);
        }

        public override void Update(AdaptationStrategy strategy)
        {
            base.Update(strategy);
        }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            List<NetworkVector> outputGradientParts = outputgradient.Segment(_repetitions);
            List<NetworkVector> inputGradientParts = new List<NetworkVector>();
            foreach (NetworkVector outputGradientPart in outputGradientParts)
            {
                inputGradientParts.Add(Weights.DotWithWeightsPerInput(outputGradientPart));
            }
            return NetworkVector.Concatenate(inputGradientParts);
        }

        public override void BackPropagate(NetworkVector outputgradient)
        {
            _biasesGradientAccumulator.Add(BiasesGradient(outputgradient));
            _weightsGradientAccumulator.Add(WeightsGradient(outputgradient));
        }
        #endregion       


    }


    //public class LayerBank : NetworkComponent
    //{
    //    #region proteced attributes
    //    int _bankWidth;
    //    int _numberOfLayerInputs;
    //    protected Layer _layer;
    //    protected NetworkVector _input;
    //    protected NetworkVector _output;
    //    protected NetworkVector _inputGradient;
    //    #endregion

    //    #region public methods
    //    public override int NumberOfInputs { get { return _bankWidth * _layer.NumberOfInputs; } }

    //    public override int NumberOfOutputs { get { return _bankWidth * _layer.NumberOfOutputs; } }

    //    public override NetworkVector Output { get { return _output; } }

    //    public override NetworkVector InputGradient { get { return _inputGradient; } }
    //    #endregion


    //    #region constructors
    //    public LayerBank(Layer layer, int width)
    //    {
    //        if (layer == null)
    //            throw new ArgumentException("Attempt to create a LayerBank with a null layer.");

    //        if (width < 1)
    //            throw new ArgumentException("Attempt to create a LayerBank with < 1 banks.");

    //        _bankWidth = width;
    //        _layer = layer;
    //        _numberOfLayerInputs = layer.NumberOfInputs;

    //        _output = new NetworkVector(_bankWidth * _layer.NumberOfOutputs);
    //        _inputGradient = new NetworkVector(_bankWidth * _layer.NumberOfInputs);
    //    }
    //    #endregion

    //    #region public methods
    //    public override void Run(NetworkVector input)
    //    {
    //        if (input.Dimension != NumberOfInputs)
    //            throw new ArgumentException("The dimension of the input does not match this WeightedCombiner.");

    //        _input = input;

    //        List<NetworkVector> inputParts = input.Segment(_bankWidth);
    //        List<NetworkVector> outputParts = new List<NetworkVector>();
    //        foreach (NetworkVector inputPart in inputParts)
    //        {
    //            _layer.Run(inputPart);
    //            outputParts.Add(_layer.Output);
    //        }
    //        _output = NetworkVector.Concatenate(outputParts);
    //    }

    //    public override void BackPropagate(NetworkVector outputgradient)
    //    {
    //        if (outputgradient.Dimension != NumberOfOutputs)
    //            throw new ArgumentException("The dimension of outputgradient does not match this WeightedCombiner.");

    //        List<NetworkVector> inputGradients = new List<NetworkVector>();
    //        List<NetworkVector> outputGradientParts = outputgradient.Segment(_bankWidth);
    //        foreach (NetworkVector outputGradientPart in outputGradientParts)
    //        {
    //            _layer.BackPropagate(outputGradientPart);
    //            inputGradients.Add(_layer.InputGradient);
    //        }
    //        _inputGradient = NetworkVector.Concatenate(inputGradients);
    //    }
    //    #endregion

    //}
}
