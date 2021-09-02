void test()
{

  float q1_min = 0;
  float q1_max = 1.0;
  float q1_step = 0.1;
  
  for(float q1=q1_min; q1<=(q1_max+q1_step); q1 = q1 + q1_step )
    {
      cout << q1 << endl;
    }

}
