package com.walking.standardofwalking.api;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

@RestController
public class AnalysisController {
    //걸음걸이 분석
    @GetMapping("/python")
    @ResponseBody
    public String pythonProcessbuilder() throws IOException, InterruptedException {
        System.out.println("pythonbuilder ");
        String arg1;
        ProcessBuilder builder;
        BufferedReader br;

        arg1 = "C:/Users/gram/Desktop/prac/liveP.py";//파이썬 파일 경로
        builder = new ProcessBuilder("python",arg1);

        builder.redirectErrorStream(true);
        Process process = builder.start();

        // 자식 프로세스가 종료될 때까지 기다림
        int exitval = process.waitFor();

        //// 서브 프로세스가 출력하는 내용을 받기 위해
        br = new BufferedReader(new InputStreamReader(process.getInputStream(),"UTF-8"));

        String line;
        while ((line = br.readLine()) != null) {
            System.out.println(">>>  " + line); // 표준출력에 쓴다
        }

        if(exitval !=0){
            //비정상종료
            System.out.println("비정상종료");
        }

        return "pythonconnet";
    }
}
