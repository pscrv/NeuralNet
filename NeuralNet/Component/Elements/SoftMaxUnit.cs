using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet2
{
    public class SoftMaxUnit : Component

    {
        #region private attributes
        private VectorBatch _input;
        protected VectorBatch _output;
        #endregion

        #region NetComponent overrides

        protected override VectorBatch _backPropagate(VectorBatch outputGradient)
        {
            if (outputGradient == null || outputGradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to the number of units.");

            return VectorBatch.ApplyFunction((x, y) => x * (1 - x) * y, _output, outputGradient);
        }
        #endregion


        #region constructors
        public SoftMaxUnit(int numberofunits)
            : base (numberofunits, numberofunits)
        {
            _input = new DataVector(numberofunits);
            _output = new DataVector(numberofunits);
        }
        #endregion

        #region public methods
        protected override VectorBatch _run(VectorBatch inputbatch)
        {
            if (inputbatch == null || inputbatch.Dimension != NumberOfInputs)
                throw new ArgumentException("input may not be null and must have dimension equal to the number of units.");

            _input = inputbatch;

            VectorBatch result = inputbatch.SubractVectorMaxima();
            result = VectorBatch.ApplyFunction(x => Math.Exp(x), result);
            result = result.DivideByComponentSums();

            _output = result;
            return _output;
        }
        #endregion
    }
}
