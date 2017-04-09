using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class NeuralFunction : NetComponent
    {
        public static double __Logistic(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input));
        }

        public static double __LogisticDerivative(double input, double output)
        {
            return output * (1 - output);
        }


        #region protected attributes
        protected int _numberOfUnits;
        protected NetworkVector _inputVector;
        protected VectorBatch _inputBatch;
        protected NetworkVector _outputVector;
        protected VectorBatch _outputBatch;
        protected ActivationFunction _neuralFunction;
        protected DerivativeFunction _neuralFunctionDerivative;
        #endregion

        #region NetworkComponent overrides
        public override int NumberOfInputs { get { return _numberOfUnits; } }
        public override int NumberOfOutputs { get { return _numberOfUnits; } }


        public override NetworkVector Run(NetworkVector inputvalues)
        {
            _inputVector = inputvalues;
            _inputBatch = null;
            _outputBatch = null;

            if (_neuralFunction != null)
            {
                _outputVector =  NetworkVector.ApplyFunctionComponentWise(inputvalues.Copy(), x => _neuralFunction(x));
            }
            else
            {
                _outputVector = inputvalues.Copy();
            }

            return _outputVector;
        }

        public override VectorBatch Run(VectorBatch inputbatch)
        {
            _inputVector = null;
            _inputBatch = inputbatch;
            _outputVector = null;

            if (_neuralFunction != null)
            {
                _outputBatch = new VectorBatch( inputbatch.AsMatrix().Map(x => _neuralFunction(x)) );
            }
            else
            {
                _outputBatch = inputbatch;
            }

            return _outputBatch;
        }


        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            if (_neuralFunctionDerivative == null)
                return outputgradient.Copy();

            return NetworkVector.ApplyFunctionComponentWise(_inputVector, _outputVector, (x, y) => _neuralFunctionDerivative(x, y));

            //NetworkVector derivative = NetworkVector.ApplyFunctionComponentWise(_inputVector, _outputVector, (x, y) => _neuralFunctionDerivative(x, y));
            //NetworkVector result = NetworkVector.ApplyFunctionComponentWise(derivative, outputgradient, (x, y) => x * y);
            //return result;
        }

        public NetworkVector InputGradient(NetworkVector outputgradient, NetworkVector input, NetworkVector output)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            if (_neuralFunctionDerivative == null)
                return outputgradient.Copy();

            return NetworkVector.ApplyFunctionComponentWise(_inputVector, _outputVector, (x, y) => _neuralFunctionDerivative(x, y));

            //NetworkVector derivative = NetworkVector.ApplyFunctionComponentWise(_inputVector, _outputVector, (x, y) => _neuralFunctionDerivative(x, y));
            //NetworkVector result = NetworkVector.ApplyFunctionComponentWise(derivative, outputgradient, (x, y) => x * y);
            //return result;
        }

        public override VectorBatch InputGradient(VectorBatch outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to NumberOfNeurons.");

            if (_neuralFunctionDerivative == null)
                return outputgradient;

            return new VectorBatch(
                _inputBatch.AsMatrix().Map2((x, y) => _neuralFunctionDerivative(x, y), _outputBatch.AsMatrix())
                );

            //VectorBatch derivative = new VectorBatch(
            //    _inputBatch.AsMatrix().Map2((x, y) => _neuralFunctionDerivative(x, y), _outputBatch.AsMatrix())
            //    );

            //VectorBatch result = new VectorBatch(
            //    derivative.AsMatrix().Map2((x, y) => x * y, outputgradient.AsMatrix() )
            //    );

            //return result;
        }
        #endregion
        

        #region Constructors
        public NeuralFunction(int numberofunits)
        {
            _numberOfUnits = numberofunits;
            _neuralFunction = null;
            _inputVector = new NetworkVector(numberofunits);
            _inputBatch = null;
            _outputVector = new NetworkVector(numberofunits);
        }

        public NeuralFunction(int numberOfUnits, ActivationFunction activationfunction, DerivativeFunction derivativefunction, double[] biases = null)
            : this(numberOfUnits)
        {
            if (activationfunction != null && derivativefunction == null)
                throw new ArgumentException("derivativefunction cannot be null, if activation is not null");

            _neuralFunction = activationfunction;
            _neuralFunctionDerivative = derivativefunction;
        }
        #endregion
    }
}
