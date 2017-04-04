using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class SoftMaxUnit : NetComponent

    {
        #region private attributes
        private int _numberOfUnits;
        private NetworkVector _input;
        protected NetworkVector _output;
        #endregion

        #region NetComponent overrides
        public override int NumberOfInputs { get { return _numberOfUnits; } }
        public override int NumberOfOutputs { get { return _numberOfUnits; } }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to the number of units.");

            return NetworkVector.ApplyFunctionComponentWise(_output, outputgradient, (x, y) => x * (1 - x) * y);
        }

        public NetworkVector InputGradient(NetworkVector outputgradient, NetworkVector output)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to the number of units.");

            return NetworkVector.ApplyFunctionComponentWise(output, outputgradient, (x, y) => x * (1 - x) * y);
        }

        public override VectorBatch InputGradient(VectorBatch outputgradients)
        {
            if (outputgradients == null || outputgradients.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to the number of units.");

            return new VectorBatch(
                (_output.AsMatrix().Map2((x, y) => x * (1 - x) * y, outputgradients.AsMatrix()))
                );
        }
        #endregion


        #region constructors
        public SoftMaxUnit(int numberofunits)
        {
            if (numberofunits <= 0)
                throw new ArgumentException("Cannot make a softmax unit will fewer then one unit.");

            _numberOfUnits = numberofunits;
            _input = new NetworkVector(numberofunits);
            _output = new NetworkVector(numberofunits);
        }

        public SoftMaxUnit()
            : this(1) { }
        #endregion

        #region public methods
        public override NetworkVector Run(NetworkVector inputvalues)
        {
            if (inputvalues == null || inputvalues.Dimension != _numberOfUnits)
                throw new ArgumentException("inputvalues may not be null and must have dimension equal to the number of units.");
            

            double max = inputvalues.Vector.Max();
            NetworkVector intermediateVector = NetworkVector.ApplyFunctionComponentWise(inputvalues.Copy(), x => Math.Exp(x - max));

            double sum = intermediateVector.SumValues();
            return NetworkVector.ApplyFunctionComponentWise(intermediateVector, x => x / sum);
        }

        public override VectorBatch Run(VectorBatch inputbatch)
        {
            if (inputbatch == null || inputbatch.Dimension != _numberOfUnits)
                throw new ArgumentException("inputvalues may not be null and must have dimension equal to the number of units.");

            double max;
            double sum;
            Matrix<double> result = Matrix<double>.Build.DenseOfMatrix(inputbatch.AsMatrix());
            foreach (Vector<double> row in result.EnumerateRows())
            {
                max = row.Max();
                row.Map(x => Math.Exp(x - max));

                sum = row.Sum();
                row.Map(x => x / sum);
            }

            return new VectorBatch(result);
        }
        #endregion
    }
}
