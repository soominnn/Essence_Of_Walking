package com.walking.standardofwalking;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;

//spring Security 구동 유무
@SpringBootApplication/*(exclude = SecurityAutoConfiguration.class)*/
public class StandardOfWalkingApplication {

    public static void main(String[] args) {
        SpringApplication.run(StandardOfWalkingApplication.class, args);
    }

}
