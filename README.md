CUDA_DynamicProgrammingExample7
===============================

another DP example in CUDA

A larger capacity CUDA implementation of the Top Coder problem 'CombinationLockDiv2':

http://community.topcoder.com/stat?c=problem_statement&pm=12969&rd=15840&rm=&cr=23177326


Same basic idea as the rest, but not optimal implementation due to uncoalesced global memory accesses.
Test configuration:

GTX 1080

CUDA 8

Windows 7 x64

Intel i7 4820k 4.5 GHz

32 GB DDR3 


____
<table>
<tr>
    <th>Number of Dials</th><th>CPU time</th><th>GPU time</th><th>CUDA Speedup</th>
</tr>

  <tr>
    <td>100</td><td> 1258 ms</td><td>  21 ms</td><td> 59.9 x</td>
  </tr>
  <tr>
    <td>200</td><td> 31438 ms</td><td>  375 ms</td><td> 20.76x</td>
  </tr>
</table>  
___  

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-60172288-1', 'auto');
  ga('send', 'pageview');

</script>
