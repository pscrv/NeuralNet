using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNet
{
    //public class WeightedCombinerBank : WeightedCombiner
    //{
    //    #region proteced attributes
    //    protected int _repetitions;
    //    protected List<NetworkVector> _inputParts;
    //    protected NetworkVector _output;
    //    #endregion

    //    #region public methods
    //    public override int NumberOfInputs { get { return _repetitions *_weights.NumberOfInputs; } }
    //    public override int NumberOfOutputs { get { return _repetitions * _weights.NumberOfOutputs; } }
    //    public override NetworkVector Output { get { return _output; } }
    //    public override NetworkVector InputGradient { get { return _inputGradient; } }
    //    #endregion


    //    #region constructors
    //    public WeightedCombinerBank(WeightedCombiner combiner, int repetitions)
    //        : base(combiner)
    //    {
    //        if (repetitions < 1)
    //            throw new ArgumentException("Attempt to create a LayerBank with < 1 banks.");

    //        _repetitions = repetitions;
    //        _inputs = new NetworkVector(NumberOfInputs);
    //        _inputParts = _inputs.Segment(_repetitions);
    //        _output = new NetworkVector(NumberOfOutputs);
    //        _inputGradient = new NetworkVector(NumberOfInputs);         
    //    }
    //    #endregion

    //    #region public methods
    //    public override void Run(NetworkVector inputvalues)
    //    {
    //        if (inputvalues.Dimension != NumberOfInputs)
    //            throw new ArgumentException("Input dimension does not match this Layer.");

    //        _inputs = inputvalues;
    //        _inputParts = inputvalues.Segment(_repetitions);            
    //        List<NetworkVector> outputParts = new List<NetworkVector>();

    //        foreach (NetworkVector inputPart in _inputParts)
    //        {
    //            outputParts.Add(_biases.SumWith(_weights.LeftMultiply(inputPart)));
    //        }
    //        _output = NetworkVector.Concatenate(outputParts);
    //    }

    //    public override void BackPropagate(NetworkVector outputgradient)
    //    {
    //        if (outputgradient.Dimension != NumberOfOutputs)
    //            throw new ArgumentException("The dimension of outputgradient does not match this WeightedCombiner.");

    //        List<NetworkVector> outputGradientParts = outputgradient.Segment(_repetitions);

    //        _setInputGradient(outputGradientParts);
    //        _updateBiases(outputGradientParts);
    //        _uptdateWeights(outputGradientParts);
    //    }


    //    #endregion

    //    #region private methods
    //    private void _uptdateWeights(List<NetworkVector> outputGradientParts)
    //    {
    //        var partArrays =
    //            _inputParts.Zip(outputGradientParts, (a, b) => new { InputPart = a.ToArray(), OutputGradientPart = b.ToArray() });

    //        double[,] weightsGradientMatrix = new double[_weights.NumberOfOutputs, _weights.NumberOfInputs];

    //        foreach (var pair in partArrays)
    //            for (int i = 0; i < _weights.NumberOfOutputs; i++)
    //                for (int j = 0; j < _weights.NumberOfInputs; j++)
    //                {
    //                    weightsGradientMatrix[i, j] += pair.OutputGradientPart[i] * pair.InputPart[j];
    //                }
    //        _weights.Subtract(new NetworkMatrix(weightsGradientMatrix));
    //    }

    //    private void _updateBiases(List<NetworkVector> outputGradientParts)
    //    {
    //        NetworkVector biasesGradient = NetworkVector.Sum(outputGradientParts);
    //        _biases.Subtract(biasesGradient);
    //    }

    //    private void _setInputGradient(List<NetworkVector> outputGradientParts)
    //    {
    //        List<NetworkVector> inputGradientParts = new List<NetworkVector>();
    //        foreach (NetworkVector outputGradientPart in outputGradientParts)
    //        {
    //            inputGradientParts.Add(_weights.DotWithWeightsPerInput(outputGradientPart));
    //        }
    //        _inputGradient = NetworkVector.Concatenate(inputGradientParts);
    //    }
    //    #endregion


    //}


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
