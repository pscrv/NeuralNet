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
        public override NetworkVector Input { get { return _input; } set { _input = value; } }
        public override NetworkVector Output { get { return _output; } protected set { _output = value; } }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to the number of units.");

            return NetworkVector.ApplyFunctionComponentWise(Output, outputgradient, (x, y) => x * (1 - x) * y);
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
        // Should subtract max input from all inputs
        // Subraction of a constant does not affect the output
        // but it will affect the exponential  - keep the oututs smallish.
        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues == null || inputvalues.Dimension != _numberOfUnits)
                throw new ArgumentException("inputvalues may not be null and must have dimension equal to the number of units.");
            
            Input = inputvalues;

            NetworkVector intermediateVector = NetworkVector.ApplyFunctionComponentWise(Input, x => Math.Exp(x));

            double sum = intermediateVector.SumValues();
            Output = NetworkVector.ApplyFunctionComponentWise(intermediateVector, x => x / sum);
        }
        #endregion
    }
}
