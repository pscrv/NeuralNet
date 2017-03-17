using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    // TODO: adapt to the new structure
    public class SoftMaxUnit : NetComponent

    {
        #region private attributes
        private int _numberOfUnits;

        NetworkVector _inputs;
        NetworkVector _outputs;
        #endregion

        #region NetComponent overrides
        public override int NumberOfInputs { get { return _numberOfUnits; } }
        public override int NumberOfOutputs { get { return _numberOfUnits; } }
        public override NetworkVector Output { get { return _outputs; } protected set { } }

        public override NetworkVector InputGradient(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to the number of units.");

            return NetworkVector.ApplyFunctionComponentWise(_outputs, outputgradient, (x, y) => x * (1 - x) * y);
        }
        #endregion


        #region constructors
        public SoftMaxUnit(int numberofunits)
        {
            if (numberofunits <= 0)
                throw new ArgumentException("Cannot make a softmax unit will fewer then one unit.");

            _numberOfUnits = numberofunits;
            _inputs = new NetworkVector(_numberOfUnits);
            _outputs = new NetworkVector(_numberOfUnits);
        }

        public SoftMaxUnit()
            : this(1) { }
        #endregion

        #region public methods
        public override void Run(NetworkVector inputvalues)
        {
            if (inputvalues == null || inputvalues.Dimension != _numberOfUnits)
                throw new ArgumentException("inputvalues may not be null and must have dimension equal to the number of units.");

            _inputs = inputvalues;
            _outputs = NetworkVector.ApplyFunctionComponentWise(_inputs, x => Math.Exp(x));

            double sum = _outputs.SumValues();
            _outputs = NetworkVector.ApplyFunctionComponentWise(_outputs, x => x / sum);
        }
        #endregion
    }
}
