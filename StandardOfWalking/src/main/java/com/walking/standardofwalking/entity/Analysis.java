package com.walking.standardofwalking.entity;

import lombok.*;
import javax.persistence.*;
import java.time.LocalDateTime;

@Getter //멤버 변수 getter 메소드 생성
@Setter
@Entity //테이블과 매핑되는 엔티티 클래스임
@NoArgsConstructor(access = AccessLevel.PROTECTED)  //클래스의 기본 생성자를 생성해줌
@ToString
public class Analysis {

    @Id //primary key설정
    @GeneratedValue(strategy = GenerationType.AUTO) //자동 증가 적용
    private Long cid;

    @Column(length=20, nullable = true)
    private String userid;

    @Column(length=20, nullable = false)
    private String leftAngle;

    @Column(length=20, nullable = false)
    private String rightAngle;

    @Column(length=20, nullable = false)
    private String leftShoulder;

    @Column(length=20, nullable = false)
    private String rightShoulder;

    @Column(length=20, nullable = false)
    private String leftHip;

    @Column(length=20, nullable = false)
    private String rightHip;
    @Column
    private LocalDateTime time;


    @Builder
    public Analysis(String userid, String leftAngle, String rightAngle, String leftShoulder, String rightShoulder, String leftHip, String rightHip, LocalDateTime time){
        this.userid = userid;
        this.leftAngle = leftAngle;
        this.rightAngle = rightAngle;
        this.leftShoulder = leftShoulder;
        this.rightShoulder = rightShoulder;
        this.leftHip = leftHip;
        this.rightHip = rightHip;
        this.time = time;
    }

}
