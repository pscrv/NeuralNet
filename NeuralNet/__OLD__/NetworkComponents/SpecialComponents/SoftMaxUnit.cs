using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class SoftMaxUnit : NetworkComponent

    {
        #region private attributes
        private int _numberOfUnits;

        NetworkVector _inputs;
        NetworkVector _outputs;
        NetworkVector _inputGradient;
        #endregion

        #region public properties
        public override int NumberOfInputs { get { return _numberOfUnits; } }
        public override int NumberOfOutputs { get { return _numberOfUnits; } }
        public override NetworkVector InputGradient { get { return _inputGradient; } }
        public override NetworkVector Output { get { return _outputs; } }
        #endregion


        #region constructors
        public SoftMaxUnit(int numberofunits)
        {
            if (numberofunits <= 0)
                throw new ArgumentException("Cannot make a softmax unit will fewer then one unit.");

            _numberOfUnits = numberofunits;
            _inputs = new NetworkVector(_numberOfUnits);
            _outputs = new NetworkVector(_numberOfUnits);
            _inputGradient = new NetworkVector(_numberOfUnits);
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


        public override void BackPropagate(NetworkVector outputgradient)
        {
            if (outputgradient == null || outputgradient.Dimension != NumberOfOutputs)
                throw new ArgumentException("outputgradient may not be null and must have dimension equal to the number of units.");

            _inputGradient = NetworkVector.ApplyFunctionComponentWise(_outputs, outputgradient, (x, y) => x * (1 - x) * y);
        }

        #endregion
    }
}
