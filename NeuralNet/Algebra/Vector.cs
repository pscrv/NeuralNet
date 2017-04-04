using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using MathNet.Numerics.LinearAlgebra;

namespace NeuralNet2
{
    public class DataVector : VectorBatch
    {

        public DataVector(VectorBatch batch)
            : base (batch)
        {
            if (batch.Count != 1)
                throw new ArgumentException("Can create a Vector only from a non-empty, singleton VectorBatch");
        }

    }
}
