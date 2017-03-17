using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public delegate double ActivationFunction(double summedInput);

    public delegate double DerivativeFunction(double input, double output);
        
}
