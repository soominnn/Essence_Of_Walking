package com.walking.standardofwalking.api;

import com.walking.standardofwalking.entity.Analysis;
import com.walking.standardofwalking.repository.AnalysisRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

@RestController
public class AnalysisController {
    //걸음걸이 분석
    @Autowired
    private AnalysisRepository analysisRepository;
    
    @GetMapping("/python/{userid}")
    @ResponseBody
    public String pythonProcessbuilder(@PathVariable String userid) throws IOException, InterruptedException {
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

        int count = 0;
        String line;
        String[] data;
        while ((line = br.readLine()) != null) {
            System.out.println(">>>  " + line);// 표준출력에 쓴다
            data = line.split("   ");
            if(count!=0)
                analysisRepository.save(new Analysis(userid,data[0],data[1],data[2],data[3],data[4],data[5],LocalDateTime.now()));
            count++;
        }

        if(exitval !=0){
            //비정상종료
            System.out.println("비정상종료");
        }

        return "pythonconnet";
    }

    @GetMapping("/result/{userid}")
    public Stream<Analysis> getResult(@PathVariable String userid){
        return analysisRepository.findAll().stream()
                .filter(m->m.getUserid().equals(userid));
    }
}
