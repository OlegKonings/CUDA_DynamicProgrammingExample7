CUDA_DynamicProgrammingExample7
===============================

another DP example in CUDA

A larger capacity CUDA implementation of the Top Coder problem 'CombinationLockDiv2':

http://community.topcoder.com/stat?c=problem_statement&pm=12969&rd=15840&rm=&cr=23177326


Same basic idea as the rest, but not optimal implementation due to uncoalesced global memory accesses.

____
<table>
<tr>
    <th>Number of Dials</th><th>CPU time</th><th>GPU time</th><th>CUDA Speedup</th>
</tr>

  <tr>
    <td>100</td><td> 1195 ms</td><td>  67 ms</td><td> 17.83x</td>
  </tr>
  <tr>
    <td>200</td><td> 36233 ms</td><td>  1745 ms</td><td> 20.76x</td>
  </tr>
</table>  
___  
