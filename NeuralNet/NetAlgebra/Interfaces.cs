using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet
{
    public interface IVectorData
    {
        Matrix<double> AsMatrix();
    }
}
