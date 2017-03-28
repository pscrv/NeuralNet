using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

using Algebra;
using System.Collections.Generic;

namespace AlgebraTests
{
    [TestClass]
    public class VectorTests
    {
        [TestMethod]
        public void CanMake_zero()
        {
            try
            {
                Vector vector = new Vector(3);
            }
            catch (Exception e)
            {
                Assert.Fail("Exception thrown by Vector constructor. Message: " + e.Message);
            }            
        }

        [TestMethod]
        public void CanMake_basis()
        {
            try
            {
                Vector vector = new Vector(5.0, 1, 2);
            }
            catch (Exception e)
            {
                Assert.Fail("Exception thrown by Vector constructor. Message: " + e.Message);
            }
        }

        [TestMethod]
        public void CanMake_full()
        {
            try
            {
                Vector vector = new Vector( new double[] { 1, 2, 3 }) ;
            }
            catch (Exception e)
            {
                Assert.Fail("Exception thrown by Vector constructor. Message: " + e.Message);
            }
        }

        [TestMethod]
        public void CanMake_unit()
        {
            try
            {
                Vector vector = Vector.MakeUnitVector(0, 1);
            }
            catch (Exception e)
            {
                Assert.Fail("Exception thrown by Vector constructor. Message: " + e.Message);
            }
        }

        [TestMethod]
        public void CanMake_composite()
        {
            try
            {
                Vector vector1 = Vector.MakeUnitVector(0, 1);
                Vector vector2 = new Vector(3);
                Vector vector3 = new Vector(new double[] { 1, 2, 3 });
                Vector composite = Vector.MakeCompositeVector(new List<Vector> { vector1, vector2, vector3 } );

            }
            catch (Exception e)
            {
                Assert.Fail("Exception thrown by Vector constructor. Message: " + e.Message);
            }
        }
        
        [TestMethod]
        public void Length()
        {
            Vector unit = Vector.MakeUnitVector(0, 17);
            Vector basis = new Vector(17);
            Vector full = new Vector(new double[17]);
            Vector composite = Vector.MakeCompositeVector(new List<Vector> { unit, basis, full });

            Assert.AreEqual(17, unit.Length);
            Assert.AreEqual(17, basis.Length);
            Assert.AreEqual(17, full.Length);
            Assert.AreEqual(51, composite.Length);
        }

        [TestMethod]
        public void ToArray()
        {
            Vector unit = Vector.MakeUnitVector(0, 17);
            Vector basis = new Vector(17);
            Vector full = new Vector(new double[17]);
            Vector composite = Vector.MakeCompositeVector(new List<Vector> { unit, basis, full });

            for (int i = 0; i < 17; i++)
            {
                Assert.AreEqual(unit[i], unit.ToArray()[i]);
                Assert.AreEqual(basis[i], basis.ToArray()[i]);
                Assert.AreEqual(full[i], full.ToArray()[i]);
            }

            for (int i = 0; i < 51; i++)
            {
                Assert.AreEqual(composite[i], composite.ToArray()[i]);
            }
        }

        [TestMethod]
        public void Add()
        {
            Vector unit = Vector.MakeUnitVector(0, 17);
            Vector zero = Vector.MakeZeroVector(17);
            Vector basis = new Vector(1, 0, 17);
            Vector full = new Vector(new double[17]);
            Vector composite = Vector.MakeCompositeVector(new List<Vector> { unit });
            

            unit.Add(zero);
            unit.Add(basis);
            unit.Add(full);
            unit.Add(composite);

            Vector expected = Vector.MakeBasisVector(3, 0, 17);
            Assert.AreEqual(expected, unit);
        }

        [TestMethod]
        public void Scale()
        {
            Vector unit = Vector.MakeUnitVector(0, 17);
            Vector zero = Vector.MakeZeroVector(17);
            Vector basis = new Vector(1, 0, 17);
            Vector full = new Vector(new double[17]);
            Vector composite = Vector.MakeCompositeVector(new List<Vector> { unit });


            unit.Scale(1.0);
            zero.Scale(13.0);
            basis.Scale(2.0);
            full.Scale(3.0);
            composite.Scale(5.0);
            
            Assert.AreEqual(1.0, unit[0]);
            for (int i = 1; i < unit.Length; i++)
            {
                Assert.AreEqual(0.0, unit[i]);
            }

            for (int i = 0; i < zero.Length; i++)
            {
                Assert.AreEqual(0.0, zero[i]);
            }

            Assert.AreEqual(2.0, basis[0]);
            for (int i = 1; i < basis.Length; i++)
            {
                Assert.AreEqual(0.0, basis[i]);
            }

            Assert.AreEqual(Vector.MakeZeroVector(17), full);

            Assert.AreEqual(5.0, composite[0]);
            for (int i = 1; i < composite.Length; i++)
            {
                Assert.AreEqual(0.0, composite[i]);
            }
        }

        [TestMethod]
        public void Subtract()
        {
            Vector unit = Vector.MakeUnitVector(0, 17);
            Vector zero = Vector.MakeZeroVector(17);
            Vector basis = new Vector(1, 0, 17);
            Vector full = new Vector(new double[17]);
            Vector composite = Vector.MakeCompositeVector(new List<Vector> { unit });


            unit.Subtract(zero);
            unit.Subtract(basis);
            unit.Subtract(full);
            unit.Subtract(composite);

            Vector expected = Vector.MakeBasisVector(-1, 0, 17);
            Assert.AreEqual(expected, unit);
        }

    }
}
