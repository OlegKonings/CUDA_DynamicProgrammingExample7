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

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-43459430-1', 'github.com');
  ga('send', 'pageview');

</script>


[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/34ae2f5d510630ae597a0fdf3187a7fe "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_DynamicProgrammingExample7)
