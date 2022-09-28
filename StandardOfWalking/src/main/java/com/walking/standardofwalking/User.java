package com.walking.standardofwalking;

import lombok.*;

import javax.persistence.*;
import javax.persistence.Id;
import java.time.LocalDateTime;

@Getter //멤버 변수 getter 메소드 생성
@Entity //테이블과 매핑되는 엔티티 클래스임
@NoArgsConstructor(access = AccessLevel.PROTECTED)  //클래스의 기본 생성자를 생성해줌
public class User {
    @Id //primary key설정
    @GeneratedValue(strategy = GenerationType.AUTO) //자동 증가 적용
    private Long cid;
    @Column(length=100, nullable = false)
    private String name;
    @Column(length=20, nullable = true)
    private String userid;
    @Column(length=200, nullable = false)
    private String password;
    @Column(length=200, nullable = false)
    private String email;
    @Column(length=10, nullable = false)
    private String gender;
    @Column(length=10, nullable = false)
    private String age;

//    private LocalDateTime createdTime;

    @Builder
    public User(String name, String userid, String password, String email,
                String gender, String age){
//        this.cid = cid;
        this.name = name;
        this.userid = userid;
        this.password = password;
        this.email = email;
        this.gender = gender;
        this.age = age;
    }
}
